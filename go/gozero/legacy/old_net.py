"""The original project's AZNet (copied so old checkpoints load without the old repo).

Input encoding (17 planes, absolute colors):
  planes 2i, 2i+1 : black / white stones, i = 0 oldest ... 7 current board
  plane 16        : 1 if black to move
Value output is tanh, from the side to move's perspective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + x)


class AZNet(nn.Module):
    def __init__(self, in_planes=17, channels=128, blocks=14, board_size=9, num_groups=8):
        super().__init__()
        self.board_size = board_size
        self.conv_in = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False)
        self.gn_in = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.blocks = nn.Sequential(*[ResidualBlock(channels, num_groups) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_gn = nn.GroupNorm(num_groups=1, num_channels=2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_gn = nn.GroupNorm(num_groups=1, num_channels=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.gn_in(self.conv_in(x)))
        out = self.blocks(out)
        p = F.relu(self.policy_gn(self.policy_conv(out))).flatten(1)
        p = self.policy_fc(p)
        v = F.relu(self.value_gn(self.value_conv(out))).flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return p, v


def load_legacy(path, device="cpu"):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    channels = sd["conv_in.weight"].shape[0]
    blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    board_size = int(round((sd["policy_fc.weight"].shape[0] - 1) ** 0.5))
    net = AZNet(channels=channels, blocks=blocks, board_size=board_size)
    net.load_state_dict(sd)
    return net.to(device).eval()
