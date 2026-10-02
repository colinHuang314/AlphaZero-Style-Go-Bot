"""Replay buffer as preallocated numpy ring arrays.

Samples are stored once (no stored augmentations); a random symmetry is applied
to each sample when a batch is drawn. Feature planes are bit-packed to keep
500k samples well under 1 GB.

The usable window grows with the total amount of data generated (as in KataGo),
so early noisy games age out quickly while later training uses a longer history.
"""
import os

import numpy as np

from ..go import symmetry
from ..go.features import NUM_PLANES


class ReplayBuffer:
    def __init__(self, capacity, n, planes=NUM_PLANES):
        self.capacity = capacity
        self.n = n
        self.planes = planes
        N = n * n
        self.packed_len = (planes * N + 7) // 8
        self.features = np.zeros((capacity, self.packed_len), dtype=np.uint8)
        self.pi = np.zeros((capacity, N + 1), dtype=np.float16)
        self.z = np.zeros(capacity, dtype=np.int8)
        self.q = np.zeros(capacity, dtype=np.float16)   # root value of the full search (z for older data)
        self.ownership = np.zeros((capacity, N), dtype=np.int8)
        self.score = np.zeros(capacity, dtype=np.float32)
        self.aux_weight = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.total_added = 0

    def add(self, d):
        k = len(d["z"])
        idx = (self.ptr + np.arange(k)) % self.capacity
        self.features[idx] = np.packbits(d["features"].reshape(k, -1), axis=1)
        self.pi[idx] = d["pi"]
        self.z[idx] = d["z"]
        self.q[idx] = d["q"] if "q" in d else d["z"]
        self.ownership[idx] = d["ownership"]
        self.score[idx] = d["score"]
        self.aux_weight[idx] = d["aux_weight"]
        self.ptr = (self.ptr + k) % self.capacity
        self.size = min(self.capacity, self.size + k)
        self.total_added += k

    def recent_indices(self, window):
        """Indices of the most recent `window` samples."""
        w = min(window, self.size)
        return (self.ptr - 1 - np.arange(w)) % self.capacity

    def sample(self, batch_size, window, rng, augment=True):
        w = min(window, self.size)
        pick = (self.ptr - 1 - rng.integers(0, w, size=batch_size)) % self.capacity
        return self.gather(pick, rng if augment else None)

    def gather(self, idx, rng=None):
        n, N, P = self.n, self.n * self.n, self.planes
        feats = np.unpackbits(self.features[idx], axis=1, count=P * N).reshape(len(idx), P, n, n)
        pi = self.pi[idx].astype(np.float32)
        own = self.ownership[idx].astype(np.float32)
        if rng is not None:
            ks = rng.integers(0, 8, size=len(idx))
            for k in range(1, 8):
                m = ks == k
                if m.any():
                    feats[m] = symmetry.transform_planes(feats[m], k)
                    pi[m] = symmetry.transform_policy(pi[m], k, n)
                    own[m] = symmetry.transform_flat(own[m], k, n)
        return {"features": feats, "pi": pi, "z": self.z[idx].astype(np.float32),
                "q": self.q[idx].astype(np.float32), "ownership": own, "score": self.score[idx], "aux_weight": self.aux_weight[idx]}

    # ------------------------------------------------------------ persistence
    def save(self, path):
        tmp = path + ".tmp.npz"
        np.savez(tmp, features=self.features[:self.size], pi=self.pi[:self.size], z=self.z[:self.size],
                 q=self.q[:self.size], ownership=self.ownership[:self.size], score=self.score[:self.size],
                 aux_weight=self.aux_weight[:self.size],
                 meta=np.array([self.ptr, self.size, self.total_added, self.capacity, self.n, self.planes]))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, capacity=None):
        d = np.load(path)
        ptr, size, total, cap, n, planes = [int(x) for x in d["meta"]]
        buf = cls(capacity or cap, n, planes)
        arrays = {name: d[name] for name in ("features", "pi", "z", "ownership", "score", "aux_weight")}
        arrays["q"] = d["q"] if "q" in d.files else d["z"]  # buffers saved before root q was recorded
        if buf.capacity != cap:
            # re-linearize oldest -> newest, then keep the newest that fit
            order = (ptr - size + np.arange(size)) % cap if size == cap else np.arange(size)
            keep = order[-buf.capacity:]
            for name, a in arrays.items():
                getattr(buf, name)[:len(keep)] = a[keep]
            buf.size = len(keep)
            buf.ptr = buf.size % buf.capacity
        else:
            for name, a in arrays.items():
                getattr(buf, name)[:size] = a
            buf.size, buf.ptr = size, ptr
        buf.total_added = total
        return buf


def window_size(total_added, min_window, max_window, alpha=0.75, beta=0.4):
    """KataGo-style growing window: starts at min_window, grows sublinearly with data."""
    if total_added <= min_window:
        return min_window
    x = total_added / min_window
    w = min_window * (1 + beta * (x ** alpha - 1) / alpha)
    return int(min(max_window, w))
