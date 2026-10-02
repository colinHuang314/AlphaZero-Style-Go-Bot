"""Referee a match between two GTP programs (our bot, GNU Go, KataGo, ...).

    python tools/gtp_match.py \
        --a "python -u -m gozero.gtp --model runs/eval_night5/candidate_c392.pt --visits 400" \
        --b "gnugo --mode gtp --chinese-rules --capture-all-dead --level 10" \
        --pairs 50 --sgf runs/gtp/c392_vs_gnugo

Same fairness rules as tools/match.py: games come in pairs from one random
2-move opening with colors swapped. The referee keeps its own board with the
training rules (area scoring, positional superko, no suicide), so an illegal
move from either program loses the game. Games end on two passes, a
resignation or --max-moves; scoring is Tromp-Taylor, so dead stones must be
captured (as on CGOS). With an integer komi a draw counts as half a win.

--time S gives each side S seconds of sudden-death time: the referee sends
time_settings / time_left and a side that runs out loses, like on CGOS.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from gozero.eval.arena import elo_diff, random_openings, wilson_interval  # noqa: E402
from gozero.go.state import GoState, move_to_str  # noqa: E402
from gozero.gtp import GTPError, parse_vertex  # noqa: E402
from gozero.selfplay.selfplay import GameRecord, to_sgf  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLOR = {1: "b", -1: "w"}


class Engine:
    """One GTP program running as a subprocess."""

    def __init__(self, cmd, stderr_path=None):
        self.cmd = cmd
        self.err = open(stderr_path, "w", encoding="utf-8") if stderr_path else subprocess.DEVNULL
        self.p = subprocess.Popen(shlex.split(cmd, posix=os.name != "nt"), cwd=ROOT, text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, bufsize=1)
        self.name = self.send("name", check=False) or cmd.split()[0]

    def send(self, command, check=True):
        self.p.stdin.write(command + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited: {self.cmd} (last command: {command})")
            line = line.rstrip("\r\n")
            if not line and lines:
                break
            if line:
                lines.append(line)
        resp = "\n".join(lines)
        if resp.startswith("?"):
            if check:
                raise RuntimeError(f"{self.name}: '{command}' failed: {resp}")
            return None
        return resp[1:].strip()

    def close(self):
        try:
            self.send("quit", check=False)
        except (RuntimeError, OSError):
            pass
        self.p.wait(timeout=10)
        if self.err is not subprocess.DEVNULL:
            self.err.close()


def play_game(black, white, opening, n, komi, max_moves, clock):
    """Returns (moves, score or None, winner color or 0 for a draw, reason)."""
    players = {1: black, -1: white}
    for e in players.values():
        e.send(f"boardsize {n}")
        e.send("clear_board")
        e.send(f"komi {komi:g}")
        if clock:
            e.send(f"time_settings {clock:g} 0 0", check=False)
    s = GoState.new(n, komi, max_moves)
    moves = []
    for m in opening:
        for e in players.values():
            e.send(f"play {COLOR[s.to_play]} {move_to_str(m, n)}")
        moves.append(m)
        s = s.play(m)
    left = {1: clock, -1: clock}
    while not s.is_terminal():
        c = s.to_play
        me, other = players[c], players[-c]
        if clock:
            for col in (1, -1):
                me.send(f"time_left {COLOR[col]} {max(0.0, left[col]):.1f} 0", check=False)
        t = time.time()
        reply = me.send(f"genmove {COLOR[c]}")
        if clock:
            left[c] -= time.time() - t
            if left[c] < 0:
                return moves, None, -c, f"{COLOR[c].upper()} lost on time"
        if reply.lower() == "resign":
            return moves, None, -c, f"{COLOR[c].upper()} resigned"
        try:
            m = parse_vertex(reply, n)
        except GTPError:
            return moves, None, -c, f"{COLOR[c].upper()} sent bad move {reply!r}"
        if not s.legal_mask()[m]:
            return moves, None, -c, f"{COLOR[c].upper()} played illegal {reply}"
        other.send(f"play {COLOR[c]} {move_to_str(m, n)}")
        moves.append(m)
        s = s.play(m)
    score = s.score()
    reason = "two passes" if s.passes >= 2 else f"move cap {max_moves}"
    return moves, score, (1 if score > 0 else -1 if score < 0 else 0), reason


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="command line of program A")
    ap.add_argument("--b", required=True, help="command line of program B")
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--board", type=int, default=9)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--opening-moves", type=int, default=2)
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--time", type=float, default=None, help="sudden-death seconds per side")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sgf", default=None, help="directory for game records and engine stderr logs")
    args = ap.parse_args()

    if args.sgf:
        os.makedirs(args.sgf, exist_ok=True)
    logp = (lambda k: os.path.join(args.sgf, f"engine_{k}.log")) if args.sgf else (lambda k: None)
    a, b = Engine(args.a, logp("a")), Engine(args.b, logp("b"))
    print(f"A = {a.name}: {args.a}\nB = {b.name}: {args.b}", flush=True)

    rng = np.random.default_rng(args.seed)
    openings = random_openings(args.pairs, args.board, args.opening_moves, rng)
    pts = {"a": 0.0, "b": 0.0}
    a_black_pts = a_black_games = 0
    lengths, reasons = [], {}
    t0 = time.time()
    g = 0
    try:
        for op in openings:
            for a_color in (1, -1):
                black, white = (a, b) if a_color == 1 else (b, a)
                moves, score, winner, reason = play_game(black, white, op, args.board, args.komi,
                                                         args.max_moves, args.time)
                a_pts = 0.5 if winner == 0 else float(winner == a_color)
                pts["a"] += a_pts
                pts["b"] += 1 - a_pts
                if a_color == 1:
                    a_black_games += 1
                    a_black_pts += a_pts
                lengths.append(len(moves))
                kind = reason.split(" ", 1)[1] if reason[0] in "BW" and " " in reason else reason
                reasons[kind] = reasons.get(kind, 0) + 1
                g += 1
                result = ("draw" if winner == 0 else
                          f"{'B' if winner == 1 else 'W'}+{abs(score):g}" if score is not None else
                          f"{'B' if winner == 1 else 'W'}+R")
                print(f"  game {g}: A as {'B' if a_color == 1 else 'W'}, {result} ({reason}, {len(moves)} moves)"
                      f" -> A {pts['a']:g} - B {pts['b']:g}", flush=True)
                if args.sgf:
                    rec = GameRecord(moves=moves, samples=[], score=score if score is not None else 0.0,
                                     winner=winner or -1, resigned=score is None)
                    sgf = to_sgf(rec, args.board, args.komi, f"B={black.name} W={white.name} {result} ({reason})")
                    if winner == 0:
                        sgf = sgf.replace("RE[W+0.0]", "RE[0]")
                    with open(os.path.join(args.sgf, f"game_{g:04d}.sgf"), "w") as f:
                        f.write(sgf)
    finally:
        a.close()
        b.close()

    games = g
    lo, hi = wilson_interval(pts["a"], games)
    wr = pts["a"] / max(1, games)
    print(f"{a.name} vs {b.name}: {pts['a']:g}-{pts['b']:g} ({100 * wr:.1f}%, 95% CI {100 * lo:.1f}-{100 * hi:.1f}%, "
          f"Elo {elo_diff(wr):+.0f} [{elo_diff(lo):+.0f}, {elo_diff(hi):+.0f}]) | "
          f"{a.name} as black {a_black_pts:g}/{a_black_games} | avg len {np.mean(lengths):.0f} | "
          + ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items())))
    print(f"{games} games in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
