"""GTP (Go Text Protocol) front end, so other programs and servers can play the bot.

    python -m gozero.gtp --model runs/eval_night5/candidate_c392.pt --visits 400

Speaks GTP v2 on stdin/stdout; diagnostics go to stderr. Rules are the same as
in training: area scoring, positional superko, suicide illegal. `komi` changes
the scoring (the net itself always assumes the training komi of 7.5).

Thinking time: each move searches up to --visits. When the controller sends
time_left (CGOS, OGS, or tools/gtp_match.py with --time), the search also stops
at a per-move budget taken from the remaining clock: what's left minus a 30 s
reserve, spread over the moves expected to remain (a 120-move game, and never
fewer than 20 more of our own moves), minus 0.3 s for latency.

--time-manage adds (see GTPEngine._search_managed):
  * with --tm-use u, the visit target comes from the clock: the visits a fresh
    search reaches in u x budget at the measured speed (reused visits count);
  * early stop: end the search once the leading move can't be overtaken in the
    visits / time left, which picks the same move sooner and banks the time;
  * hard moves: if at the normal limit the top two moves are close, the best-Q
    move isn't the most visited, or the evaluation swung since our last move,
    keep searching up to --tm-ext times the visits and time;
  * decided games: with a win estimate above 97% or below 3%, use half the visits
    and never extend.
Moves that are the same up to a symmetry of the position (e.g. the four 3-4
points on an empty board) are pooled into one candidate for all of these tests.
"""
import argparse
import os
import sys
import time

import numpy as np

from .go.state import GoState, default_komi, move_to_str, str_to_move
from .go.symmetry import invariant_symmetries, permutations
from .mcts.mcts import MCTSConfig, Node, pick_move, search_single

COLORS = {"b": 1, "black": 1, "w": -1, "white": -1}
COMMANDS = ["protocol_version", "name", "version", "known_command", "list_commands", "quit",
            "boardsize", "clear_board", "komi", "play", "genmove", "kgs-genmove_cleanup", "undo",
            "showboard", "final_score", "time_settings", "kgs-time_settings", "time_left"]


class GTPError(Exception):
    pass


def parse_vertex(s, n):
    """GTP vertex ("D4", "pass") -> flat move index, rejecting points off the board."""
    s = s.strip().upper()
    if s == "PASS":
        return n * n
    cols = "ABCDEFGHJKLMNOPQRST"[:n]
    if len(s) < 2 or s[0] not in cols or not s[1:].isdigit() or not 1 <= int(s[1:]) <= n:
        raise GTPError("invalid coordinate")
    return str_to_move(s, n)


def move_classes(state, moves):
    """Class id per entry of `moves`: moves that map onto each other under a symmetry
    of the position share an id (the smallest index in their orbit). The previous
    board must be symmetric too, so a ko can't make two "equal" moves differ."""
    n = state.rules.n
    ks = invariant_symmetries(state.board, n)
    if state.prev is not None and len(ks) > 1:
        ks = [k for k in ks if k in invariant_symmetries(state.prev.board, n)]
    cls = np.arange(len(moves))
    if len(ks) == 1:
        return cls
    pos = {int(m): i for i, m in enumerate(moves)}
    perms = permutations(n)
    for i, m in enumerate(moves):
        if m < n * n:  # pass is its own class
            cls[i] = min(pos[int(perms[k][m])] for k in ks if int(perms[k][m]) in pos)
    return cls


def class_stats(root, cls):
    """Pooled visits and Q per symmetry class, plus the closeness / best-Q tests."""
    ids = np.unique(cls)
    N = np.array([root.Nc[cls == c].sum() for c in ids])
    W = np.array([root.Wc[cls == c].sum() for c in ids])
    order = np.argsort(-N, kind="stable")
    top = order[0]
    n1 = N[top]
    n2 = N[order[1]] if len(order) > 1 else 0.0
    Q = np.where(N > 0, W / np.maximum(N, 1), -np.inf)
    eligible = np.flatnonzero(N >= max(1.0, 0.1 * n1))      # ignore barely-searched moves
    qbest = eligible[np.argmax(Q[eligible])] if len(eligible) else top
    return dict(ids=ids, N=N, top=ids[top], lead=n1 - n2,
                close=n1 > 0 and n2 >= 0.5 * n1,
                mismatch=qbest != top and Q[qbest] > Q[top] + 0.02)


class GTPEngine:
    """Game state, search tree and clock for one GTP session.

    `evaluator` is any MCTS evaluator (a NetEvaluator in normal use, a
    UniformEvaluator in tests). Search follows the official evaluation:
    no Dirichlet noise, greedy choice of the most-visited move.
    """

    # main-time clock plan: spread what's left over the moves expected to remain
    HORIZON = 120          # assumed game length in moves (both colors)
    MIN_MOVES = 20         # always plan for at least this many more of our own moves
    RESERVE = 30.0         # seconds of main time never planned for (network lag in long games)
    LAG = 0.3              # seconds per move assumed for network / process latency

    # time-management settings (--time-manage)
    TM_EASY_Q = 0.94       # |root Q| above this (win estimate > 97% or < 3%) = decided game
    TM_EASY_FRAC = 0.5     # ... which gets this fraction of the visits and no extension
    TM_SWING = 0.15        # root Q change since our previous move that counts as a surprise
    TM_CHUNK = 128         # visits between checks

    def __init__(self, evaluator, n=9, visits=400, max_time=None, resign=None,
                 leaf_batch=16, max_moves=1000, name="go-zero", log=None, seed=0,
                 time_manage=False, tm_ext=3.0, tm_use=None):
        self.ev = evaluator
        self.n = n
        self.visits = visits
        self.max_time = max_time          # cap in seconds per move (None = visits only)
        self.time_manage = time_manage
        self.tm_ext = tm_ext              # hard moves: up to this many times the visits and time
        self.tm_use = tm_use              # visit target from the clock: what a fresh search reaches in tm_use x budget
        self.rate = None                  # measured search speed, visits per second (moving average)
        self.resign = resign              # resign when the best move's Q is below this (None = never)
        self.leaf_batch = leaf_batch
        self.max_moves = max_moves
        self.name = name
        self.log = log or (lambda msg: None)
        self.cfg = MCTSConfig(dirichlet_eps=0.0)
        self.rng = np.random.default_rng(seed)
        self.komi = default_komi(n)
        self.time_left = {1: None, -1: None}   # (seconds, stones) per color from time_left
        self.quit = False
        self.new_game()

    # ------------------------------------------------------------- game state
    def new_game(self):
        self.log(f"new game (komi {self.komi:g})")
        self.state = GoState.new(self.n, self.komi, self.max_moves)
        self.history = []
        self.tree = Node(self.state)
        self.last_q = {1: None, -1: None}  # root Q after each color's previous genmove (swing test)

    def _advance(self, move):
        self.history.append(self.state)
        self.state = self.state.play(move)
        t = self.tree
        self.tree = t.child_by_move(move) if t.expanded else Node(self.state)

    def _sync_turn(self, color):
        """GTP allows a color to move twice; model the skipped turn as a pass."""
        if color != self.state.to_play:
            self._advance(self.n * self.n)

    def play(self, color, move_str):
        self._sync_turn(color)
        move = parse_vertex(move_str, self.n)
        if not self.state.legal_mask()[move]:
            raise GTPError("illegal move")
        self._advance(move)

    def _avail(self, secs):
        """Main time we may plan with: all but the reserve (half the clock once it's that low)."""
        return secs - min(self.RESERVE, secs / 2)

    def move_budget(self, color):
        """Seconds to spend on this move, from the clock (None = no limit)."""
        budget = self.max_time
        left = self.time_left[color]
        if left is not None:
            secs, stones = left
            if stones > 0:
                clock = secs / stones                                  # byo-yomi period
            else:                                                      # main time
                clock = self._avail(secs) / max(self.MIN_MOVES, (self.HORIZON - self.state.move_number) // 2)
            clock = max(0.05, clock - self.LAG)
            budget = clock if budget is None else min(budget, clock)
        return budget

    def genmove(self, color):
        self._sync_turn(color)
        if self.state.is_terminal():
            return "pass"
        budget = self.move_budget(color)
        t0 = time.time()
        root = self.tree  # total visits include those reused from the previous move, as in arena matches
        reused = int(root.N)
        note = ""
        if self.time_manage:
            move, note = self._search_managed(root, budget, color, t0)
        else:
            stop = (lambda: time.time() - t0 > budget) if budget is not None else None
            search_single(root, self.ev, self.cfg, self.visits, leaf_batch=self.leaf_batch, should_stop=stop)
            if not root.expanded or root.N == 0:  # no time for even one evaluation
                search_single(root, self.ev, self.cfg, 1)
            move = pick_move(root, 0.0, self.rng)
        idx = int(np.flatnonzero(root.moves == move)[0])
        q = root.Wc[idx] / max(1.0, root.Nc[idx])
        self.log(f"move {self.state.move_number + 1}: {move_to_str(move, self.n)} "
                 f"visits {int(root.N)} ({reused} reused) q {q:+.3f} time {time.time() - t0:.2f}s"
                 + (f" (budget {budget:.2f}s)" if budget is not None else "") + note)
        if self.resign is not None and self.state.move_number >= 20 and q < self.resign:
            return "resign"
        self._advance(move)
        return move_to_str(move, self.n).lower()

    def _search_managed(self, root, budget, color, t0):
        """Search with early stopping and hard-move extension; returns (move, log note).

        Visit target T: with --tm-use u and a clock, T = (measured visits/s) x u x budget,
        i.e. what a fresh search reaches in u x budget, capped by --visits. Reused visits
        count toward T: strength grows with the log of the total, so every move aiming at
        the same total is the best split of the clock, and a move whose tree was mostly
        searched already banks the time for later moves (whose budgets, and so targets,
        then rise). Without --tm-use or a clock, T = --visits.

        Two phases. "normal": up to T (half in a decided game) and u x budget seconds.
        "extended" (hard moves): up to tm_ext times both, but never more than 1/6 of the
        remaining plannable clock, and it ends once the move is no longer hard. In either
        phase the search stops as soon as the leading symmetry class can't be caught in
        the visits that remain (by count, or by time at the current speed), unless the
        position is still hard.
        """
        if not root.expanded:
            search_single(root, self.ev, self.cfg, 1)
        cls = move_classes(root.state, root.moves)
        secs = self.time_left[color][0] if self.time_left[color] is not None else None
        u = self.tm_use if (self.tm_use is not None and budget is not None) else 1.0
        base_time = None if budget is None else u * budget
        V = self.visits
        if self.tm_use is not None and base_time is not None and self.rate is not None:
            V = min(V, max(self.TM_CHUNK, self.rate * base_time))
        ext_time = None
        if base_time is not None:
            ext_time = base_time * self.tm_ext
            if secs is not None:
                ext_time = min(ext_time, max(base_time, self._avail(secs) / 6 - self.LAG))
        n0 = root.N
        phase, why, swing = "normal", "cap", False
        while True:
            rq = root.q()
            easy = abs(rq) > self.TM_EASY_Q
            prev = self.last_q[color]
            swing = prev is not None and abs(rq - prev) > self.TM_SWING
            st = class_stats(root, cls)
            hard = not easy and (st["close"] or st["mismatch"])
            if phase == "normal" and swing and not easy:
                phase = "extended"
            if phase == "normal":
                vis_lim, time_lim = V * (self.TM_EASY_FRAC if easy else 1.0), base_time
            else:
                vis_lim, time_lim = V * self.tm_ext, ext_time
            t = time.time() - t0
            # early stop: the leader can't be overtaken in what's left
            left = vis_lim - root.N
            if time_lim is not None and t > 0.05 and root.N > n0:
                left = min(left, (root.N - n0) / t * (time_lim - t))
            if not hard and st["lead"] > left:
                why = "decided"
                break
            if root.N >= vis_lim or (time_lim is not None and t >= time_lim):
                if phase == "normal" and hard:
                    phase = "extended"
                    continue
                why = "cap" if root.N >= vis_lim else "time"
                break
            if phase == "extended" and root.N >= V and not hard and (not swing or root.N >= 2 * V):
                why = "resolved"
                break
            stop = (lambda lim=time_lim: time.time() - t0 > lim) if time_lim is not None else None
            search_single(root, self.ev, self.cfg, min(vis_lim, root.N + self.TM_CHUNK),
                          leaf_batch=self.leaf_batch, should_stop=stop)
        # play the most visited class; within it, its most visited move
        st = class_stats(root, cls)
        members = np.flatnonzero(cls == st["top"])
        move = int(root.moves[members[np.argmax(root.Nc[members])]])
        self.last_q[color] = root.q()
        t = time.time() - t0
        if t > 0.2 and root.N - n0 >= 200:  # update the speed estimate from this move's fresh visits
            r = (root.N - n0) / t
            self.rate = r if self.rate is None else 0.7 * self.rate + 0.3 * r
        flags = [f for f, on in (("easy", easy), ("close", st["close"]), ("mismatch", st["mismatch"]),
                                 ("swing", swing)) if on]
        sym = len(np.unique(cls)) < len(cls)
        return move, (f" [target {int(V)}, {phase}, stop: {why}" + (", " + "+".join(flags) if flags else "")
                      + (", symmetric" if sym else "") + "]")

    def undo(self):
        if not self.history:
            raise GTPError("cannot undo")
        self.state = self.history.pop()
        self.tree = Node(self.state)

    def final_score(self):
        s = self.state.score()
        if s == 0:
            return "0"
        return f"B+{s:g}" if s > 0 else f"W+{-s:g}"

    # --------------------------------------------------------------- protocol
    def handle(self, line):
        """Process one command line; returns the full response (or None for blank lines)."""
        line = line.lstrip("﻿").split("#", 1)[0].strip()  # a byte-order mark (e.g. from a PowerShell pipe)
        if not line:
            return None
        parts = line.split()
        cid = ""
        if parts[0].isdigit():
            cid, parts = parts[0], parts[1:]
        if not parts:
            return None
        cmd, args = parts[0].lower(), parts[1:]
        try:
            out = self._dispatch(cmd, args)
            return f"={cid} {out}".rstrip() + "\n\n"
        except GTPError as e:
            return f"?{cid} {e}\n\n"
        except (ValueError, IndexError, KeyError) as e:
            return f"?{cid} syntax error ({e})\n\n"

    def _dispatch(self, cmd, args):
        if cmd == "protocol_version":
            return "2"
        if cmd == "name":
            return self.name
        if cmd == "version":
            return "1.0"
        if cmd == "known_command":
            return "true" if args and args[0].lower() in COMMANDS else "false"
        if cmd == "list_commands":
            return "\n".join(COMMANDS)
        if cmd == "quit":
            self.quit = True
            return ""
        if cmd == "boardsize":
            if int(args[0]) != self.n:
                raise GTPError("unacceptable size")
            self.new_game()
            return ""
        if cmd == "clear_board":
            self.new_game()
            return ""
        if cmd == "komi":
            self.komi = float(args[0])
            if self.history:  # keep the game; komi only changes scoring
                moves = [s.last_moves[0] for s in self.history[1:]] + [self.state.last_moves[0]]
                self.new_game()
                for m in moves:
                    self._advance(m)
            else:
                self.new_game()
            return ""
        if cmd == "play":
            if args[1].lower() == "resign":
                return ""
            self.play(COLORS[args[0].lower()], args[1])
            return ""
        if cmd in ("genmove", "kgs-genmove_cleanup"):
            return self.genmove(COLORS[args[0].lower()])
        if cmd == "undo":
            self.undo()
            return ""
        if cmd == "showboard":
            return "\n" + repr(self.state)
        if cmd == "final_score":
            return self.final_score()
        if cmd in ("time_settings", "kgs-time_settings"):
            self.time_left = {1: None, -1: None}
            return ""
        if cmd == "time_left":
            self.time_left[COLORS[args[0].lower()]] = (float(args[1]), int(args[2]))
            self.log(f"time_left {args[0].lower()} {args[1]} {args[2]}")  # vs our own move times: network lag
            return ""
        raise GTPError("unknown command")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint (GoNet or original-project AZNet)")
    ap.add_argument("--visits", type=int, default=400, help="visits per move (upper limit when a clock is used)")
    ap.add_argument("--max-time", type=float, default=None, help="seconds per move cap")
    ap.add_argument("--resign", type=float, default=None, help="resign below this Q, e.g. -0.95 (default: never)")
    ap.add_argument("--time-manage", action="store_true",
                    help="early stop on decided moves, extend hard ones (see the module docstring)")
    ap.add_argument("--tm-ext", type=float, default=3.0, help="hard moves: up to this many times the visits and time")
    ap.add_argument("--tm-use", type=float, default=None,
                    help="with a clock, aim each move at the visits a fresh search reaches in this fraction of "
                         "its time budget (reused visits count; --visits becomes an upper limit)")
    ap.add_argument("--leaf-batch", type=int, default=16)
    ap.add_argument("--name", default=None)
    ap.add_argument("--quiet", action="store_true", help="no per-move log on stderr")
    ap.add_argument("--log-file", default=None,
                    help="also append the per-move log, with timestamps, to this file (relative to the repo root)")
    args = ap.parse_args()

    import torch
    from .loop import load_anchor

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = load_anchor(path, dev)
    n = getattr(getattr(ev.model, "cfg", None), "board_size", None) or ev.model.board_size
    logf = None
    if args.log_file:
        lp = args.log_file if os.path.isabs(args.log_file) else os.path.join(root, args.log_file)
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        logf = open(lp, "a", encoding="utf-8")

    def log(m):
        if not args.quiet:
            print(m, file=sys.stderr, flush=True)
        if logf:
            logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n")
            logf.flush()
    name = args.name or "go-zero-" + os.path.basename(path).replace(".pt", "").replace("candidate_", "")
    eng = GTPEngine(ev, n=n, visits=args.visits, max_time=args.max_time, resign=args.resign,
                    leaf_batch=args.leaf_batch, name=name, log=log,
                    time_manage=args.time_manage, tm_ext=args.tm_ext, tm_use=args.tm_use)

    # warm up CUDA and the numba kernels so the first real move isn't slow on the clock
    t = time.time()
    search_single(Node(GoState.new(n)), ev, eng.cfg, 64, leaf_batch=args.leaf_batch)
    log(f"{name} ready on {dev} ({time.time() - t:.1f}s warmup), {args.visits} visits per move"
        + (f", time management on (hard moves up to {args.tm_ext:g}x"
           + (f", target from {args.tm_use:g} x clock budget" if args.tm_use else "") + ")" if args.time_manage else ""))

    for line in sys.stdin:
        resp = eng.handle(line)
        if resp is None:
            continue
        sys.stdout.write(resp)
        sys.stdout.flush()
        if eng.quit:
            break


if __name__ == "__main__":
    main()
