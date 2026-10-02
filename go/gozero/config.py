"""Run configuration: one YAML file maps onto these dataclasses."""
from dataclasses import asdict, dataclass, field, fields, is_dataclass

import yaml

from .mcts.mcts import MCTSConfig
from .nn.model import NetConfig
from .selfplay.selfplay import SelfPlayConfig
from .train.trainer import TrainConfig


@dataclass
class LoopConfig:
    cycle_samples: int = 4000        # new training samples generated per cycle
    train_ratio: float = 4.0         # samples trained on per new sample (reuse)
    min_buffer: int = 20000          # don't train before the buffer has this many samples
    min_window: int = 25000
    max_window: int = 400000
    buffer_capacity: int = 500000
    val_game_frac: float = 0.05      # fraction of games held out for validation
    val_capacity: int = 20000
    snapshot_every: int = 10         # cycles between saved model snapshots
    buffer_save_every: int = 10
    sgf_per_cycle: int = 2
    seed: int = 0
    selfplay_workers: int = 1        # >1: self-play in that many processes sharing the GPU (selfplay/workers.py)


@dataclass
class EvalConfig:
    every_cycles: int = 10
    anchors: list = field(default_factory=lambda: ["anchors/AZNET9_epoch_300.pt"])
    num_pairs: int = 32              # 64 games per anchor
    visits: int = 100
    vs_previous: bool = True         # also play the previous snapshot


@dataclass
class RunConfig:
    net: NetConfig = field(default_factory=NetConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _fill(dc_type, data):
    obj = dc_type()
    for f in fields(dc_type):
        if f.name not in (data or {}):
            continue
        val = data[f.name]
        cur = getattr(obj, f.name)
        if is_dataclass(cur):
            val = _fill(type(cur), val)
        setattr(obj, f.name, val)
    unknown = set(data or {}) - {f.name for f in fields(dc_type)}
    if unknown:
        raise ValueError(f"unknown config keys in {dc_type.__name__}: {sorted(unknown)}")
    return obj


def load_config(path):
    with open(path) as f:
        cfg = _fill(RunConfig, yaml.safe_load(f))
    # keep board size consistent everywhere
    cfg.selfplay.board_size = cfg.net.board_size
    return cfg


def dump_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
