"""Play a match between two checkpoints (new GoNet or original-project AZNet).

    python tools/match.py A.pt B.pt --pairs 200 --visits 400
    python tools/match.py runs/9x9_a/latest.pt anchors/AZNET9_epoch_300.pt --pairs 200 --visits 400 --sgf out_dir
    python tools/match.py X.pt X.pt --komi 7 --a-komi-fix --a-score-utility 0.05   # CGOS komi, engine fix on A only

This is the official evaluation: equal visits for both sides, the same MCTS
settings, colors swapped on every opening, greedy move choice.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from gozero.eval.arena import Player, play_match  # noqa: E402
from gozero.loop import load_anchor  # noqa: E402
from gozero.mcts.evaluator import ScoreUtility  # noqa: E402
from gozero.mcts.mcts import MCTSConfig  # noqa: E402
from gozero.selfplay.selfplay import GameRecord, to_sgf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--visits", type=int, default=400)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--fpu", type=float, default=0.2)
    ap.add_argument("--b-c-puct", type=float, default=None, help="different c_puct for player B")
    ap.add_argument("--opening-moves", type=int, default=2)
    ap.add_argument("--board", type=int, default=9)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--komi", type=float, default=None, help="default: the board's (7.5 on 9x9); CGOS uses 7")
    for side in ("a", "b"):
        ap.add_argument(f"--{side}-komi-fix", action="store_true", help=f"player {side.upper()}: ScoreUtility komi fix")
        ap.add_argument(f"--{side}-score-utility", type=float, default=0.0,
                        help=f"player {side.upper()}: ScoreUtility score bonus weight")
    ap.add_argument("--score-sigma", type=float, default=2.0)
    ap.add_argument("--sgf", default=None, help="directory to write game records")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_a = MCTSConfig(c_puct=args.c_puct, fpu_reduction=args.fpu, dirichlet_eps=0.0)
    cfg_b = MCTSConfig(c_puct=args.b_c_puct or args.c_puct, fpu_reduction=args.fpu, dirichlet_eps=0.0)
    name = lambda p: os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p).replace(".pt", "")
    evs = {}
    for side, path in (("a", args.a), ("b", args.b)):
        ev = load_anchor(path, dev)
        fix, w = getattr(args, f"{side}_komi_fix"), getattr(args, f"{side}_score_utility")
        if fix or w:
            ev.utility = ScoreUtility(komi_fix=fix, score_weight=w, sigma=args.score_sigma)
        evs[side] = ev
    tag = lambda side: "".join([" +komifix" if getattr(args, f"{side}_komi_fix") else "",
                                f" +score{getattr(args, f'{side}_score_utility'):g}"
                                if getattr(args, f"{side}_score_utility") else ""])
    pa = Player(name(args.a) + tag("a"), evs["a"], args.visits, cfg_a)
    pb = Player(name(args.b) + tag("b"), evs["b"], args.visits, cfg_b)
    t = time.time()
    r = play_match(pa, pb, args.pairs, args.board, args.opening_moves, args.seed, verbose=args.verbose,
                   komi=args.komi)
    print(r.summary())
    if args.komi is not None:
        for c, cname in ((1, "black"), (-1, "white")):
            g = [x for x in r.games if x[2] == c]
            print(f"  A as {cname}: {sum(x[1] == c for x in g)} wins, {sum(x[1] == 0 for x in g)} draws, "
                  f"{sum(x[1] == -c for x in g)} losses")
    print(f"{r.games_played} games in {time.time() - t:.0f}s")
    if args.sgf:
        os.makedirs(args.sgf, exist_ok=True)
        from gozero.go.state import default_komi
        for i, (moves, w, a_color, score) in enumerate(r.games):
            rec = GameRecord(moves=moves, samples=[], score=score, winner=w)
            black = pa.name if a_color == 1 else pb.name
            white = pb.name if a_color == 1 else pa.name
            with open(os.path.join(args.sgf, f"game_{i:04d}.sgf"), "w") as f:
                komi = default_komi(args.board) if args.komi is None else args.komi
                f.write(to_sgf(rec, args.board, komi, f"B={black} W={white}"))


if __name__ == "__main__":
    main()
