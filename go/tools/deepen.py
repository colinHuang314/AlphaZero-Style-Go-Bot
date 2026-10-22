"""Make a deeper network that starts out computing exactly the same function (net2net-style).

    python tools/deepen.py SRC.pt DST.pt --blocks 15 [--check runs/exp_c/val_buffer.npz]

New residual blocks are spread evenly through the tower, each with the last BatchNorm's weight and bias
set to zero: the block's branch then outputs 0 and, because every block's input is already post-ReLU,
relu(x + 0) = x, so the block is an identity until training moves it. Everything else (stem, old blocks,
heads) is copied. The loop state (cycle, games, samples) is kept; the optimizer state is not (its
shapes changed), so training restarts the lr warmup.
--check runs the old and new nets on held-out positions and reports the largest output difference.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from gozero.nn.model import GoNet, NetConfig, load_model, save_checkpoint  # noqa: E402


def deepen(model, blocks):
    old = model.cfg.blocks
    assert blocks > old
    cfg = NetConfig(**{**model.cfg.__dict__, "blocks": blocks})
    new = GoNet(cfg)
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("tower.")}
    # positions in the new tower that hold old blocks, spread so new blocks fall evenly between them
    old_pos = [round(i * (blocks - 1) / max(1, old - 1)) for i in range(old)] if old > 1 else [0]
    for i, p in enumerate(old_pos):
        for k, v in model.state_dict().items():
            if k.startswith(f"tower.{i}."):
                sd[f"tower.{p}." + k[len(f"tower.{i}."):]] = v
    fresh = new.state_dict()
    for p in range(blocks):
        if p in old_pos:
            continue
        for k, v in fresh.items():
            if k.startswith(f"tower.{p}."):
                sd[k] = v
        sd[f"tower.{p}.bn2.weight"] = torch.zeros_like(fresh[f"tower.{p}.bn2.weight"])
        sd[f"tower.{p}.bn2.bias"] = torch.zeros_like(fresh[f"tower.{p}.bn2.bias"])
    new.load_state_dict(sd)
    return new.eval(), old_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--check", default=None, help="replay buffer .npz with positions to compare outputs on")
    args = ap.parse_args()
    model, ck = load_model(args.src, "cpu")
    new, old_pos = deepen(model, args.blocks)
    extra = {k: v for k, v in ck.items() if k not in ("net_config", "model_state_dict", "trainer")}
    save_checkpoint(args.dst, new, extra)
    print(f"{args.src}: {model.cfg.blocks} blocks -> {args.dst}: {args.blocks} blocks "
          f"(old blocks at positions {old_pos}); params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"-> {sum(p.numel() for p in new.parameters()) / 1e6:.2f}M; kept {sorted(extra)}")
    if args.check:
        from gozero.train.replay import ReplayBuffer
        buf = ReplayBuffer.load(args.check)
        x = torch.from_numpy(buf.gather(buf.recent_indices(2048))["features"]).float()
        with torch.no_grad():
            a, b = model(x), new(x)
        for k in a:
            print(f"  max |difference| in {k}: {(a[k] - b[k]).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
