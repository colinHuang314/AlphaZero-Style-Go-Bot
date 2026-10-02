"""Continuous self-play / training loop (AlphaZero style, no gating).

    python -m gozero.loop --config configs/9x9.yaml --run runs/9x9_a --hours 9

Every cycle: generate `cycle_samples` new samples with the latest network,
add them to the replay buffer, then train on the buffer. Progress is measured
against fixed anchors (the original project's models) instead of gating.

Safe to stop at any time (Ctrl+C): the latest model, optimizer and replay
buffer are saved, and the next run with the same --run resumes.
"""
import argparse
import csv
import os
import shutil
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

from .config import dump_config, load_config
from .eval.arena import Player, play_match
from .go.state import default_komi
from .legacy.old_net import load_legacy
from .mcts.evaluator import LegacyEvaluator, NetEvaluator
from .mcts.mcts import MCTSConfig
from .nn.model import build, load_model, save_checkpoint
from .selfplay.selfplay import SelfPlay, record_to_training, to_sgf
from .selfplay.workers import WorkerPool
from .train.replay import ReplayBuffer, window_size
from .train.trainer import Trainer

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard optional
    SummaryWriter = None


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class MetricsLog:
    """Append-only CSV (one row per cycle, columns grow as needed) + optional TensorBoard."""

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, "metrics.csv")
        self.rows = []
        if os.path.exists(self.path):
            with open(self.path, newline="") as f:
                self.rows = list(csv.DictReader(f))
        self.tb = SummaryWriter(os.path.join(run_dir, "tb")) if SummaryWriter else None

    def write(self, row, step):
        self.rows.append({k: v for k, v in row.items()})
        keys = []
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        with open(self.path + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(self.path + ".tmp", self.path)
        if self.tb:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "cycle":
                    self.tb.add_scalar(k.replace(".", "/"), v, step)
            self.tb.flush()


def load_anchor(path, device):
    """An anchor is either an original-project checkpoint or a new GoNet checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "net_config" in ck:
        model, _ = load_model(path, device)
        return NetEvaluator(model, seed=1)
    return LegacyEvaluator(load_legacy(path, device), seed=1)


def run(args):
    cfg = load_config(args.config)
    run_dir = args.run
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "sgf"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = cfg.net.board_size
    L = cfg.loop

    latest = os.path.join(run_dir, "latest.pt")
    if os.path.exists(latest):
        model, ck = load_model(latest, device)
        state = ck["loop_state"]
        log(f"resuming {run_dir} at cycle {state['cycle']} ({state['games']} games, {state['samples']} samples)")
    else:
        torch.manual_seed(L.seed)
        model = build(cfg.net).to(device).eval()
        ck = None
        state = {"cycle": 0, "games": 0, "samples": 0, "train_samples": 0, "wall_hours": 0.0}
        log(f"new run {run_dir}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    dump_config(cfg, os.path.join(run_dir, f"config_cycle{state['cycle']}.yaml"))

    trainer = Trainer(model, cfg.train, device)
    if ck is not None and "trainer" in ck:
        trainer.load_state_dict(ck["trainer"])

    buf_path = os.path.join(run_dir, "buffer.npz")
    val_path = os.path.join(run_dir, "val_buffer.npz")
    buffer = ReplayBuffer.load(buf_path, L.buffer_capacity) if os.path.exists(buf_path) else ReplayBuffer(L.buffer_capacity, n)
    val = ReplayBuffer.load(val_path, L.val_capacity) if os.path.exists(val_path) else ReplayBuffer(L.val_capacity, n)

    rng = np.random.default_rng(L.seed + state["cycle"])
    pool = sp = None
    if L.selfplay_workers > 1:
        pool = WorkerPool(L.selfplay_workers, cfg.selfplay, cfg.net, run_dir, seed=L.seed + state["cycle"],
                          force_cpu=device.type == "cpu")
        pool.publish(model)
        log(f"self-play in {L.selfplay_workers} worker processes, "
            f"{cfg.selfplay.num_parallel // L.selfplay_workers} games each")
        ev_mark = (time.time(), 0.0)
    else:
        sp = SelfPlay(cfg.selfplay, NetEvaluator(model, seed=L.seed + state["cycle"]), seed=L.seed + state["cycle"] + 1)
    metrics = MetricsLog(run_dir)
    anchors = {os.path.basename(p).replace(".pt", ""): load_anchor(p, device) for p in cfg.eval.anchors}

    def save_all(with_buffer):
        save_checkpoint(latest, model, {"loop_state": state, "trainer": trainer.state_dict()})
        if with_buffer:
            buffer.save(buf_path)
            val.save(val_path)

    start = time.time()
    hours_at_start = state["wall_hours"]
    deadline = start + args.hours * 3600 if args.hours else None
    try:
        stop_file = os.path.join(run_dir, "STOP")
        while deadline is None or time.time() < deadline:
            if os.path.exists(stop_file):  # graceful stop requested between cycles
                os.remove(stop_file)
                log("STOP file found, stopping")
                break
            state["cycle"] += 1
            cyc = state["cycle"]
            t0 = time.time()
            # ------------------------------------------------ self-play
            new_samples, games = 0, []

            def add(d):
                nonlocal new_samples
                if d is None:
                    return
                if rng.random() < L.val_game_frac:
                    val.add(d)
                else:
                    buffer.add(d)
                    new_samples += len(d["z"])

            if pool is None:
                evals0 = sp.total_evals
                while new_samples < L.cycle_samples:
                    for rec in sp.step():
                        games.append(rec)
                        add(record_to_training(rec, n))
                t_sp = time.time() - t0
                ev_rate = (sp.total_evals - evals0) / t_sp
            else:
                while new_samples < L.cycle_samples:
                    got = pool.get()
                    if got is None:
                        continue
                    info, d = got
                    games.append(SimpleNamespace(**info))
                    add(d)
                t_sp = time.time() - t0
                # throughput over the whole cycle: workers keep playing while the main process trains
                now, ev_now = time.time(), pool.total_evals()
                ev_rate = (ev_now - ev_mark[1]) / max(1e-6, now - ev_mark[0])
                ev_mark = (now, ev_now)
            state["games"] += len(games)
            state["samples"] += new_samples
            for i, rec in enumerate(games[:L.sgf_per_cycle]):
                with open(os.path.join(run_dir, "sgf", f"c{cyc:05d}_{i}.sgf"), "w") as f:
                    f.write(to_sgf(rec, n, default_komi(n), f"cycle {cyc}"))

            lengths = np.array([len(r.moves) for r in games])
            row = {
                "cycle": cyc, "games_total": state["games"], "samples_total": state["samples"],
                "sp.games": len(games), "sp.black_winrate": float(np.mean([r.winner == 1 for r in games])),
                "sp.len_mean": float(lengths.mean()), "sp.len_p90": float(np.percentile(lengths, 90)),
                "sp.capped_frac": float(np.mean([r.capped for r in games])),
                "sp.resigned_frac": float(np.mean([r.resigned for r in games])),
                "sp.score_abs_mean": float(np.mean([abs(r.score) for r in games if not r.resigned] or [0])),
                "sp.evals_per_s": ev_rate, "sp.seconds": t_sp,
            }
            # resignation false positives: games where resigning was disabled, but a side
            # crossed the threshold and still won
            ctrl = [r for r in games if not r.resign_allowed]
            flagged = [(r, p) for r in ctrl for p, f in r.resign_would_have.items() if f]
            if flagged:
                row["sp.resign_false_pos"] = float(np.mean([r.winner == p for r, p in flagged]))

            # ------------------------------------------------ training
            t1 = time.time()
            if buffer.size >= L.min_buffer:
                win = window_size(buffer.total_added, L.min_window, L.max_window)
                steps = max(1, int(new_samples * L.train_ratio / cfg.train.batch_size))
                st = trainer.train_steps(buffer, win, steps, rng)
                state["train_samples"] += steps * cfg.train.batch_size
                row.update({f"train.{k}": v for k, v in st.items()})
                row.update({"train.steps_total": trainer.steps, "train.window": win, "train.lr": trainer.lr_now()})
                if cyc % 5 == 0 and val.size > 0:
                    row.update({f"val.{k}": v for k, v in trainer.validate(val).items()})
                if pool is not None:
                    pool.publish(model)  # in-process self-play shares `model`, so it sees updates directly
            row["train.seconds"] = time.time() - t1

            # ------------------------------------------------ snapshots / eval
            if cyc % L.snapshot_every == 0:
                save_checkpoint(os.path.join(run_dir, "models", f"model_c{cyc:05d}.pt"), model,
                                {"loop_state": dict(state)})
            if cfg.eval.every_cycles and cyc % cfg.eval.every_cycles == 0 and buffer.size >= L.min_buffer:
                t2 = time.time()
                me = Player("current", NetEvaluator(model, seed=cyc), cfg.eval.visits,
                            MCTSConfig(c_puct=cfg.selfplay.mcts.c_puct, fpu_reduction=cfg.selfplay.mcts.fpu_reduction,
                                       dirichlet_eps=0.0))
                opponents = dict(anchors)
                if cfg.eval.vs_previous:
                    prev = sorted(f for f in os.listdir(os.path.join(run_dir, "models")) if f < f"model_c{cyc:05d}")
                    if prev:
                        pm, _ = load_model(os.path.join(run_dir, "models", prev[-1]), device)
                        opponents["prev"] = NetEvaluator(pm, seed=2)
                for name, ev in opponents.items():
                    r = play_match(me, Player(name, ev, cfg.eval.visits, MCTSConfig(dirichlet_eps=0.0)),
                                   cfg.eval.num_pairs, n, seed=cyc)
                    row[f"eval.{name}.winrate"] = r.a_winrate
                    log("  " + r.summary())
                row["eval.seconds"] = time.time() - t2

            state["wall_hours"] = hours_at_start + (time.time() - start) / 3600
            row["wall_hours"] = state["wall_hours"]
            metrics.write(row, state["samples"])
            save_all(with_buffer=cyc % L.buffer_save_every == 0)
            tr = f"P {row.get('train.policy', float('nan')):.3f} V {row.get('train.value', float('nan')):.3f}" \
                 f" O {row.get('train.ownership', float('nan')):.3f} ent {row.get('train.policy_entropy', float('nan')):.2f}"
            vl = f" | val P {row['val.policy']:.3f} V {row['val.value']:.3f}" if "val.policy" in row else ""
            log(f"cycle {cyc}: {len(games)} games (len {lengths.mean():.0f}, B {100 * row['sp.black_winrate']:.0f}%, "
                f"capped {100 * row['sp.capped_frac']:.0f}%), {row['sp.evals_per_s']:.0f} ev/s, "
                f"sp {t_sp:.0f}s | {tr}{vl} | buf {buffer.size}")
    except KeyboardInterrupt:
        log("interrupted, saving...")
    finally:
        if pool is not None:
            pool.close()
        state["wall_hours"] = hours_at_start + (time.time() - start) / 3600
        save_all(with_buffer=True)
        log(f"saved. total {state['games']} games, {state['wall_hours']:.2f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = until Ctrl+C)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
