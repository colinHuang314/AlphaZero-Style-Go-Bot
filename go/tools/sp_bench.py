"""Self-play speed at different worker counts (needs the GPU to itself).

    python tools/sp_bench.py --model runs/9x9_a/latest.pt --workers 1 2 4 6 --seconds 120

For each worker count, runs self-play only (no training) with the config's games
and visits, and reports network evaluations per second after a warm-up. Use it to
pick `loop.selfplay_workers`.
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from gozero.config import load_config  # noqa: E402
from gozero.nn.model import load_model  # noqa: E402
from gozero.selfplay.workers import WorkerPool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/9x9.yaml")
    ap.add_argument("--model", default="runs/9x9_a/latest.pt")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--warmup", type=float, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, _ = load_model(args.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in args.workers:
            pool = WorkerPool(k, cfg.selfplay, cfg.net, tmp, seed=k)
            pool.publish(model)
            games = 0

            def drain(until):
                nonlocal games
                while time.time() < until:
                    if pool.get(timeout=0.5) is not None:
                        games += 1

            drain(time.time() + args.warmup)
            t0, e0, g0 = time.time(), pool.total_evals(), games
            drain(t0 + args.seconds)
            rate = (pool.total_evals() - e0) / (time.time() - t0)
            pool.close()
            results.append((k, rate, games - g0))
            print(f"{k} worker(s): {rate:,.0f} evals/s, {games - g0} games finished in {args.seconds:.0f} s", flush=True)
    base = results[0][1]
    print("\n" + "\n".join(f"  {k} worker(s): {r / base:.2f}x" for k, r, _ in results))


if __name__ == "__main__":
    main()
