"""Batched self-play.

Runs many games at once, all searched together by BatchedMCTS. Games that
finish are replaced immediately, so the generator is persistent across
training cycles; swapping in a new network just changes the evaluator
(games in progress continue with the new net, as in KataGo).

Improvements over the original loop:
  * policy targets are raw visit distributions (never temperature-sharpened)
  * playout cap randomization (KataGo): only a fraction of moves get a full
    search and become policy-training samples; the rest use a cheap search
  * move-selection temperature decays smoothly over the game
  * tree reuse between moves
  * optional resignation with a no-resign control group to measure false positives
"""
from dataclasses import dataclass, field

import numpy as np

from ..go.features import make_features
from ..go.state import GoState
from ..mcts.mcts import BatchedMCTS, MCTSConfig, Node, pick_move


@dataclass
class SelfPlayConfig:
    board_size: int = 9
    num_parallel: int = 256
    full_visits: int = 400
    fast_visits: int = 80
    full_prob: float = 0.25
    temp_start: float = 1.0
    temp_end: float = 0.25
    temp_halflife: float = 8.0       # moves (per game) for temperature to halve its distance to temp_end
    tree_reuse: bool = True
    resign_enabled: bool = False
    resign_threshold: float = -0.95  # root Q from the resigning side's view
    resign_consecutive: int = 3
    resign_disabled_frac: float = 0.2
    max_moves: int = 0               # 0 = rules default (3 * n * n)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)


@dataclass
class GameRecord:
    moves: list
    samples: list          # (features uint8, pi float32, to_play, move_number, root q)
    score: float = 0.0     # final black - white - komi
    winner: int = 0
    resigned: bool = False
    capped: bool = False
    resign_would_have: dict = field(default_factory=dict)  # player -> True if value crossed threshold
    resign_allowed: bool = False
    final_ownership: object = None


class _Slot:
    __slots__ = ("root", "record", "low_count", "resign_allowed", "resign_flag")

    def __init__(self, state, resign_allowed):
        self.root = Node(state)
        self.record = GameRecord(moves=[], samples=[])
        self.low_count = {1: 0, -1: 0}
        self.resign_allowed = resign_allowed
        self.resign_flag = {1: False, -1: False}


class SelfPlay:
    def __init__(self, cfg: SelfPlayConfig, evaluator, seed=None):
        self.cfg = cfg
        self.evaluator = evaluator
        self.rng = np.random.default_rng(seed)
        self.mcts = BatchedMCTS(cfg.mcts, self.rng)
        self.slots = [self._new_slot() for _ in range(cfg.num_parallel)]
        self.total_evals = 0

    def _new_slot(self):
        s = GoState.new(self.cfg.board_size, max_moves=self.cfg.max_moves or None)
        allow = self.cfg.resign_enabled and self.rng.random() >= self.cfg.resign_disabled_frac
        return _Slot(s, allow)

    def temperature(self, move_number):
        c = self.cfg
        return c.temp_end + (c.temp_start - c.temp_end) * 0.5 ** (move_number / c.temp_halflife)

    def step(self):
        """Play one move in every game. Returns list of finished GameRecords."""
        cfg = self.cfg
        n_act = cfg.board_size ** 2 + 1
        full = self.rng.random(len(self.slots)) < cfg.full_prob
        visits = [cfg.full_visits if f else cfg.fast_visits for f in full]
        roots = [sl.root for sl in self.slots]
        self.total_evals += self.mcts.search(roots, self.evaluator, visits, noise=list(full))

        finished = []
        for i, sl in enumerate(self.slots):
            root = sl.root
            st = root.state
            q = root.q()  # root value after the search, from the side to move's view
            if full[i]:
                sl.record.samples.append((make_features(st), root.visit_distribution(n_act),
                                          st.to_play, st.move_number, q))
            # resignation bookkeeping
            if q < cfg.resign_threshold:
                sl.low_count[st.to_play] += 1
                if sl.low_count[st.to_play] >= cfg.resign_consecutive:
                    sl.resign_flag[st.to_play] = True
            else:
                sl.low_count[st.to_play] = 0
            if sl.resign_allowed and sl.resign_flag[st.to_play]:
                self._finish(sl, winner=-st.to_play, resigned=True)
                finished.append(sl.record)
                self.slots[i] = self._new_slot()
                continue

            move = pick_move(root, self.temperature(st.move_number), self.rng)
            sl.record.moves.append(move)
            if cfg.tree_reuse:
                sl.root = root.child_by_move(move)
            else:
                sl.root = Node(st.play(move))
            if sl.root.terminal:
                self._finish(sl, winner=sl.root.state.winner(), resigned=False)
                finished.append(sl.record)
                self.slots[i] = self._new_slot()
        return finished

    def _finish(self, sl, winner, resigned):
        rec = sl.record
        st = sl.root.state
        rec.winner = winner
        rec.resigned = resigned
        rec.score = st.score()
        rec.capped = (not resigned) and st.passes < 2
        rec.resign_would_have = dict(sl.resign_flag)
        rec.resign_allowed = sl.resign_allowed
        rec.final_ownership = st.ownership()

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator


def record_to_training(rec, n):
    """Convert a finished game into training arrays (side-to-move perspective targets).

    `z` is the game result and `q` the root value of the full search at that move
    (both from the side to move's view); the trainer can blend them as the value target.

    For resigned games the ownership / score targets come from the position at
    resignation, which is noisier; they get weight 0 for those aux targets.
    """
    if not rec.samples:
        return None
    feats = np.stack([s[0] for s in rec.samples])
    pis = np.stack([s[1] for s in rec.samples]).astype(np.float32)
    to_play = np.array([s[2] for s in rec.samples], dtype=np.int8)
    z = (rec.winner * to_play).astype(np.int8)
    q = np.array([s[4] if len(s) > 4 else z[i] for i, s in enumerate(rec.samples)], dtype=np.float32)
    own = rec.final_ownership[None, :].astype(np.int8) * to_play[:, None]
    score = (rec.score * to_play).astype(np.float32)
    aux_w = np.full(len(to_play), 0.0 if rec.resigned else 1.0, dtype=np.float32)
    return {"features": feats, "pi": pis, "z": z, "q": q, "ownership": own.astype(np.int8),
            "score": score, "aux_weight": aux_w}


def to_sgf(rec, n, komi, result_note=""):
    cols = "abcdefghijklmnopqrs"
    out = [f"(;GM[1]FF[4]SZ[{n}]KM[{komi}]RU[Tromp-Taylor]"]
    res = f"{'B' if rec.winner == 1 else 'W'}+{'R' if rec.resigned else abs(rec.score)}"
    out.append(f"RE[{res}]C[{result_note}]")
    color = "B"
    for m in rec.moves:
        if m == n * n:
            out.append(f";{color}[]")
        else:
            r, c = divmod(m, n)
            out.append(f";{color}[{cols[c]}{cols[r]}]")
        color = "W" if color == "B" else "B"
    out.append(")")
    return "".join(out)
