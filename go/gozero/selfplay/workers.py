"""Self-play in several processes that share the GPU.

One Python process can't keep the GPU busy: tree search runs in Python/numba on a
single core. Self-play ran at ~3,500 evaluations/s, an estimated 10-20% of what the
4060 can evaluate for this net, and it slowed by a third when a CPU-only program
(GNU Go) ran alongside, so the CPU side is the limit. With K workers, each runs its own SelfPlay on num_parallel / K games
(so the total number of games in flight is unchanged) with its own copy of the
network, and sends finished games to the training process through a queue.

Weights: the training process writes them to `weights_path` (atomically) and bumps
`version`; workers check between moves and reload, so games in progress continue
with the new network, as in the single-process loop. Workers keep playing while the
main process trains, so a cycle's samples can come partly from the previous network
(at most one cycle old, as in KataGo's asynchronous self-play).
"""
import multiprocessing as mp
import os
import queue
import time
from dataclasses import replace

import numpy as np


def _worker(wid, sp_cfg, net_cfg, weights_path, version, evals, stop, out_q, seed, force_cpu):
    import torch

    from ..mcts.evaluator import NetEvaluator
    from ..nn.model import build
    from .selfplay import SelfPlay, record_to_training

    torch.set_num_threads(1)
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    model = build(net_cfg).to(device).eval()
    ver = -1
    n = sp_cfg.board_size

    def reload():
        for attempt in range(20):  # the file is replaced atomically, but tolerate a racing open
            try:
                model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
                return
            except (OSError, RuntimeError, EOFError):
                time.sleep(0.2)
        raise RuntimeError(f"worker {wid}: could not load {weights_path}")

    while version.value < 0 and not stop.is_set():
        time.sleep(0.1)
    ver = version.value
    reload()
    sp = SelfPlay(sp_cfg, NetEvaluator(model, seed=seed), seed=seed + 1)
    while not stop.is_set():
        if version.value != ver:
            ver = version.value
            reload()
        e0 = sp.total_evals
        for rec in sp.step():
            info = dict(moves=rec.moves, winner=rec.winner, score=rec.score, resigned=rec.resigned,
                        capped=rec.capped, resign_would_have=rec.resign_would_have,
                        resign_allowed=rec.resign_allowed, version=ver)
            out_q.put((info, record_to_training(rec, n)))
        with evals.get_lock():
            evals[wid] += sp.total_evals - e0


class WorkerPool:
    """K self-play processes. Use publish() to send weights, get() to receive games."""

    def __init__(self, k, sp_cfg, net_cfg, run_dir, seed, force_cpu=False):
        self.ctx = mp.get_context("spawn")  # CUDA needs spawned (not forked) children
        self.weights_path = os.path.join(run_dir, "selfplay_weights.pt")
        self.version = self.ctx.Value("i", -1)
        self.evals = self.ctx.Array("d", k)
        self.stop = self.ctx.Event()
        self.q = self.ctx.Queue(maxsize=4096)
        per = max(2, sp_cfg.num_parallel // k)
        self.procs = []
        for w in range(k):
            p = self.ctx.Process(target=_worker, daemon=True,
                                 args=(w, replace(sp_cfg, num_parallel=per), net_cfg, self.weights_path,
                                       self.version, self.evals, self.stop, self.q, seed + 1000 * w, force_cpu))
            p.start()
            self.procs.append(p)

    def publish(self, model):
        import torch
        tmp = self.weights_path + ".tmp"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
        os.replace(tmp, self.weights_path)
        with self.version.get_lock():
            self.version.value += 1

    def total_evals(self):
        with self.evals.get_lock():
            return float(np.sum(self.evals[:]))

    def get(self, timeout=5.0):
        """One finished game (info dict, training dict or None), or None if nothing arrived."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            dead = [i for i, p in enumerate(self.procs) if not p.is_alive()]
            if dead:
                raise RuntimeError(f"self-play worker(s) {dead} exited (exit codes "
                                   f"{[self.procs[i].exitcode for i in dead]})")
            return None

    def close(self):
        self.stop.set()
        # drain so workers blocked on a full queue can exit
        t_end = time.time() + 10
        while time.time() < t_end and any(p.is_alive() for p in self.procs):
            try:
                self.q.get(timeout=0.2)
            except queue.Empty:
                pass
        for p in self.procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
