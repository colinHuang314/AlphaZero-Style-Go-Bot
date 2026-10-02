"""Network optimization on replay-buffer batches."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    batch_size: int = 256
    optimizer: str = "sgd"          # "sgd" (Nesterov momentum, as AlphaZero) or "adamw"
    lr: float = 0.02
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_steps: int = 300
    grad_clip: float = 5.0
    policy_weight: float = 1.0
    value_weight: float = 1.0
    ownership_weight: float = 1.0
    score_weight: float = 0.5
    value_q_weight: float = 0.0     # value target = (1 - w) * game result z + w * root q of the full search
    use_amp: bool = True


def make_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 else decay).append(p)  # no weight decay on BN / biases
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)


def to_tensors(batch, device):
    t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in batch.items()}
    t["features"] = t["features"].float()
    return t


def compute_losses(model, t, cfg: TrainConfig):
    out = model(t["features"])
    logp = F.log_softmax(out["policy"].float(), dim=1)
    policy = -(t["pi"] * logp).sum(1).mean()
    vlogit = out["value_logit"].float()
    w = cfg.value_q_weight
    target = t["z"] if w == 0 or "q" not in t else (1 - w) * t["z"] + w * t["q"]
    value_target = F.binary_cross_entropy_with_logits(vlogit, (target + 1) / 2)  # what is optimized
    with torch.no_grad():  # vs the game result alone, comparable with runs before value_q_weight
        value = F.binary_cross_entropy_with_logits(vlogit, (t["z"] + 1) / 2)
    aw = t["aux_weight"]
    aw_sum = aw.sum().clamp(min=1.0)
    own = (((out["ownership"].float() - t["ownership"]) ** 2).mean(1) * aw).sum() / aw_sum
    score = (F.smooth_l1_loss(out["score"].float() / 10, t["score"] / 10, reduction="none") * aw).sum() / aw_sum
    total = (cfg.policy_weight * policy + cfg.value_weight * value_target
             + cfg.ownership_weight * own + cfg.score_weight * score)
    with torch.no_grad():
        p = logp.exp()
        stats = {
            "loss": total.item(), "policy": policy.item(), "value": value.item(),
            "value_target": value_target.item(),
            "ownership": own.item(), "score": score.item(),
            "policy_entropy": -(p * logp).sum(1).mean().item(),
            "target_entropy": -(t["pi"] * torch.log(t["pi"].clamp(min=1e-12))).sum(1).mean().item(),
            "policy_top1": (out["policy"].argmax(1) == t["pi"].argmax(1)).float().mean().item(),
            "value_acc": ((out["value_logit"] > 0).float() == (t["z"] > 0).float()).float().mean().item(),
        }
    return total, stats


class Trainer:
    def __init__(self, model, cfg: TrainConfig, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.opt = make_optimizer(model, cfg)
        self.scaler = torch.amp.GradScaler(device.type, enabled=cfg.use_amp and device.type == "cuda")
        self.steps = 0

    def lr_now(self):
        warm = min(1.0, (self.steps + 1) / max(1, self.cfg.warmup_steps))
        return self.cfg.lr * warm

    def train_steps(self, buffer, window, n_steps, rng):
        self.model.train()
        acc = {}
        for _ in range(n_steps):
            for g in self.opt.param_groups:
                g["lr"] = self.lr_now()
            t = to_tensors(buffer.sample(self.cfg.batch_size, window, rng), self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                loss, stats = compute_losses(self.model, t, self.cfg)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.steps += 1
            stats["grad_norm"] = float(gn)
            for k, v in stats.items():
                acc[k] = acc.get(k, 0.0) + v
        self.model.eval()
        return {k: v / max(1, n_steps) for k, v in acc.items()}

    @torch.no_grad()
    def validate(self, buffer, max_samples=4096):
        """Losses on held-out games (never trained on) to detect overfitting."""
        if buffer.size == 0:
            return {}
        self.model.eval()
        idx = buffer.recent_indices(max_samples)
        acc, count = {}, 0
        for i in range(0, len(idx), 1024):
            chunk = idx[i:i + 1024]
            t = to_tensors(buffer.gather(chunk), self.device)
            _, stats = compute_losses(self.model, t, self.cfg)
            for k, v in stats.items():
                acc[k] = acc.get(k, 0.0) + v * len(chunk)
            count += len(chunk)
        return {k: v / count for k, v in acc.items()}

    def state_dict(self):
        return {"opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(), "steps": self.steps}

    def load_state_dict(self, d):
        self.opt.load_state_dict(d["opt"])
        self.scaler.load_state_dict(d["scaler"])
        self.steps = d["steps"]
