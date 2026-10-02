"""Policy / value network.

A pre-activation-free ResNet (conv-BN-ReLU) like AlphaZero, with heads borrowed
from KataGo that give more learning signal per game:
  * policy: fully convolutional board logits + pass logit from global pooling
  * value:  win/loss logit (cross-entropy instead of MSE on tanh)
  * score:  final score margin from the side to move (auxiliary)
  * ownership: who owns each point at game end (auxiliary, strong signal on small data)
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..go.features import NUM_PLANES


@dataclass
class NetConfig:
    board_size: int = 9
    blocks: int = 8
    channels: int = 96
    head_channels: int = 32
    value_hidden: int = 128
    in_planes: int = NUM_PLANES


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


def global_pool(x):
    return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)


class GoNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        c, hc = cfg.channels, cfg.head_channels
        self.stem = nn.Sequential(nn.Conv2d(cfg.in_planes, c, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*[ResBlock(c) for _ in range(cfg.blocks)])

        self.p_conv = nn.Sequential(nn.Conv2d(c, hc, 1, bias=False), nn.BatchNorm2d(hc), nn.ReLU(inplace=True))
        self.p_global = nn.Linear(2 * hc, hc)  # global context added back to each point
        self.p_out = nn.Conv2d(hc, 1, 1)
        self.p_pass = nn.Linear(2 * hc, 1)

        self.v_conv = nn.Sequential(nn.Conv2d(c, hc, 1, bias=False), nn.BatchNorm2d(hc), nn.ReLU(inplace=True))
        self.v_own = nn.Conv2d(hc, 1, 1)
        self.v_fc = nn.Sequential(nn.Linear(2 * hc, cfg.value_hidden), nn.ReLU(inplace=True))
        self.v_out = nn.Linear(cfg.value_hidden, 2)  # [win logit, score]

    def forward(self, x):
        h = self.tower(self.stem(x))

        p = self.p_conv(h)
        g = global_pool(p)
        p = F.relu(p + self.p_global(g)[:, :, None, None])
        board_logits = self.p_out(p).flatten(1)
        policy_logits = torch.cat([board_logits, self.p_pass(g)], dim=1)

        v = self.v_conv(h)
        ownership = torch.tanh(self.v_own(v).flatten(1))
        vo = self.v_out(self.v_fc(global_pool(v)))
        return {
            "policy": policy_logits,        # (B, N+1)
            "value_logit": vo[:, 0],        # (B,) P(win for side to move) = sigmoid
            "score": vo[:, 1] * 10.0,       # (B,) points, side-to-move perspective
            "ownership": ownership,         # (B, N) +1 = side to move owns
        }


def build(cfg: NetConfig):
    return GoNet(cfg)


def save_checkpoint(path, model, extra=None):
    payload = {"net_config": asdict(model.cfg), "model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    import os
    os.replace(tmp, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GoNet(NetConfig(**ck["net_config"]))
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval(), ck


def count_params(model):
    return sum(p.numel() for p in model.parameters())
