"""Head-to-head matches between two players, played in parallel.

Fairness rules:
  * both players search with the same number of visits per move (equal sims)
  * games come in pairs from the same random opening with colors swapped,
    so an unbalanced opening cannot favor either player
  * moves are chosen greedily (most visits), no noise; diversity comes only
    from the openings and each evaluator's random symmetry
  * each player keeps its own search tree (with reuse across moves)
"""
import math
from dataclasses import dataclass, field

import numpy as np

from ..go.state import GoState
from ..mcts.mcts import BatchedMCTS, MCTSConfig, Node, pick_move


@dataclass
class Player:
    name: str
    evaluator: object
    visits: int = 400
    mcts: MCTSConfig = field(default_factory=lambda: MCTSConfig(dirichlet_eps=0.0))


def random_openings(num, n, k_moves, rng, avoid_edge=True):
    """`num` opening move sequences of length k_moves, uniform over non-edge points."""
    openings = []
    for _ in range(num):
        s = GoState.new(n)
        seq = []
        for _ in range(k_moves):
            legal = np.flatnonzero(s.legal_mask()[:-1])
            if avoid_edge:
                rc = np.array([divmod(int(m), n) for m in legal])
                keep = (rc[:, 0] > 0) & (rc[:, 0] < n - 1) & (rc[:, 1] > 0) & (rc[:, 1] < n - 1)
                legal = legal[keep] if keep.any() else legal
            m = int(rng.choice(legal))
            seq.append(m)
            s = s.play(m)
        openings.append(seq)
    return openings


@dataclass
class MatchResult:
    a_name: str
    b_name: str
    a_wins: int = 0
    b_wins: int = 0
    a_wins_as_black: int = 0
    a_games_as_black: int = 0
    lengths: list = field(default_factory=list)
    capped: int = 0
    games: list = field(default_factory=list)  # (moves, winner, a_color, score)

    @property
    def games_played(self):
        return self.a_wins + self.b_wins

    @property
    def a_winrate(self):
        return self.a_wins / max(1, self.games_played)

    def summary(self):
        lo, hi = wilson_interval(self.a_wins, self.games_played)
        elo, elo_lo, elo_hi = elo_diff(self.a_winrate), elo_diff(lo), elo_diff(hi)
        return (f"{self.a_name} vs {self.b_name}: {self.a_wins}-{self.b_wins} "
                f"({100 * self.a_winrate:.1f}%, 95% CI {100 * lo:.1f}-{100 * hi:.1f}%, "
                f"Elo {elo:+.0f} [{elo_lo:+.0f}, {elo_hi:+.0f}]) | "
                f"{self.a_name} as black {self.a_wins_as_black}/{self.a_games_as_black} | "
                f"avg len {np.mean(self.lengths):.0f}, capped {self.capped}")


def wilson_interval(wins, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def elo_diff(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / p - 1)


def play_match(a: Player, b: Player, num_pairs, n, opening_moves=2, seed=0, verbose=False):
    """Play 2 * num_pairs games (each opening once with each color assignment)."""
    rng = np.random.default_rng(seed)
    openings = random_openings(num_pairs, n, opening_moves, rng)
    mcts = {"a": BatchedMCTS(a.mcts, rng), "b": BatchedMCTS(b.mcts, rng)}
    res = MatchResult(a.name, b.name)

    games = []  # dict per game
    for g, op in enumerate(openings):
        for a_color in (1, -1):
            s = GoState.new(n)
            for m in op:
                s = s.play(m)
            games.append({"state": s, "a_color": a_color, "moves": list(op),
                          "trees": {"a": Node(s), "b": Node(s)}})

    def to_move(i):
        return "a" if games[i]["state"].to_play == games[i]["a_color"] else "b"

    same_search = a.mcts == b.mcts and a.visits == b.visits
    live = list(range(len(games)))
    while live:
        if same_search:
            # one search over all games, each root with its own side's evaluator (bigger GPU batches)
            keys = [to_move(i) for i in live]
            roots = [games[i]["trees"][k] for i, k in zip(live, keys)]
            evs = [(a if k == "a" else b).evaluator for k in keys]
            mcts["a"].search(roots, evs, a.visits)
            for i, root in zip(live, roots):
                games[i]["pending_move"] = pick_move(root, 0.0, rng)
        else:
            for key, p in (("a", a), ("b", b)):
                idx = [i for i in live if to_move(i) == key]
                if not idx:
                    continue
                roots = [games[i]["trees"][key] for i in idx]
                mcts[key].search(roots, p.evaluator, p.visits)
                for i, root in zip(idx, roots):
                    games[i]["pending_move"] = pick_move(root, 0.0, rng)
        still = []
        for i in live:
            gm = games[i]
            m = gm.pop("pending_move")
            gm["moves"].append(m)
            gm["state"] = gm["state"].play(m)
            for pid, t in gm["trees"].items():
                gm["trees"][pid] = t.child_by_move(m) if t.expanded else Node(gm["state"])
            if gm["state"].is_terminal():
                st = gm["state"]
                w = st.winner()
                a_won = w == gm["a_color"]
                res.a_wins += a_won
                res.b_wins += not a_won
                if gm["a_color"] == 1:
                    res.a_games_as_black += 1
                    res.a_wins_as_black += a_won
                res.lengths.append(st.move_number)
                res.capped += st.passes < 2
                res.games.append((gm["moves"], w, gm["a_color"], st.score()))
                gm["trees"] = None
                if verbose:
                    print(f"  game {len(res.games)}: {res.a_name if a_won else res.b_name} wins "
                          f"(score {st.score():+.1f}, {st.move_number} moves) -> {res.a_wins}-{res.b_wins}",
                          flush=True)
            else:
                still.append(i)
        live = still
    return res
