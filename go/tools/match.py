"""Play a match between two checkpoints (new GoNet or original-project AZNet).

    python tools/match.py A.pt B.pt --pairs 200 --visits 400
    python tools/match.py runs/9x9_a/latest.pt anchors/AZNET9_epoch_300.pt --pairs 200 --visits 400 --sgf out_dir

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
    ap.add_argument("--sgf", default=None, help="directory to write game records")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_a = MCTSConfig(c_puct=args.c_puct, fpu_reduction=args.fpu, dirichlet_eps=0.0)
    cfg_b = MCTSConfig(c_puct=args.b_c_puct or args.c_puct, fpu_reduction=args.fpu, dirichlet_eps=0.0)
    name = lambda p: os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p).replace(".pt", "")
    pa = Player(name(args.a), load_anchor(args.a, dev), args.visits, cfg_a)
    pb = Player(name(args.b), load_anchor(args.b, dev), args.visits, cfg_b)
    t = time.time()
    r = play_match(pa, pb, args.pairs, args.board, args.opening_moves, args.seed, verbose=args.verbose)
    print(r.summary())
    print(f"{r.games_played} games in {time.time() - t:.0f}s")
    if args.sgf:
        os.makedirs(args.sgf, exist_ok=True)
        from gozero.go.state import default_komi
        for i, (moves, w, a_color, score) in enumerate(r.games):
            rec = GameRecord(moves=moves, samples=[], score=score, winner=w)
            black = pa.name if a_color == 1 else pb.name
            white = pb.name if a_color == 1 else pa.name
            with open(os.path.join(args.sgf, f"game_{i:04d}.sgf"), "w") as f:
                f.write(to_sgf(rec, args.board, default_komi(args.board), f"B={black} W={white}"))


if __name__ == "__main__":
    main()
