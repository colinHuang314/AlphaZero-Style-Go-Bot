"""Batched neural-network evaluators used by MCTS.

An evaluator turns a list of GoStates into (priors over legal moves, value) pairs.
Values are always from the perspective of the side to move, in [-1, 1].

`submit(states)` launches the GPU work and returns a callable that blocks for the
result, so MCTS can do CPU work for other games while the GPU is busy.
"""
import numpy as np
import torch

from ..go import symmetry
from ..go.features import make_features


def batch_priors(logits, states):
    """Masked softmax for a batch; returns list of (legal moves, priors)."""
    legal = np.stack([s.legal_mask() for s in states]).astype(bool)
    x = np.where(legal, logits.astype(np.float64), -np.inf)
    x = np.exp(x - x.max(axis=1, keepdims=True))
    x /= x.sum(axis=1, keepdims=True)
    out = []
    for i in range(len(states)):
        moves = np.flatnonzero(legal[i])
        out.append((moves, x[i, moves].astype(np.float32)))
    return out


class TorchEvaluator:
    """Shared batching / symmetry / mixed-precision logic.

    random_symmetry: evaluate each batch under a random board symmetry (as in AlphaZero),
    which averages out any orientation bias across searches.
    """

    def __init__(self, model, n, device=None, random_symmetry=True, use_amp=True, seed=None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.random_symmetry = random_symmetry
        self.use_amp = use_amp and self.device.type == "cuda"
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.num_evals = 0

    # subclass hooks -------------------------------------------------------
    def features(self, states):
        raise NotImplementedError

    def _outputs(self, model_out):
        """Map model output to a dict of tensors."""
        raise NotImplementedError

    def _values(self, out):
        raise NotImplementedError

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _launch(self, states, k):
        x = self.features(states)
        if k:
            x = symmetry.transform_planes(x, k)
        xt = torch.from_numpy(x).to(self.device, non_blocking=True).float()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
            return self._outputs(self.model(xt))

    def _finish(self, gpu_out, k, count):
        out = {key: v.float().cpu().numpy() for key, v in gpu_out.items()}
        if k:
            inv = symmetry.inverse(k)
            out["policy"] = symmetry.transform_policy(out["policy"], inv, self.n)
            if "ownership" in out:
                out["ownership"] = symmetry.transform_flat(out["ownership"], inv, self.n)
        self.num_evals += count
        return out

    def raw(self, states, k=0):
        """Network outputs in original board orientation (numpy), under symmetry k."""
        return self._finish(self._launch(states, k), k, len(states))

    def submit(self, states):
        k = int(self.rng.integers(8)) if self.random_symmetry else 0
        gpu_out = self._launch(states, k)

        def result():
            out = self._finish(gpu_out, k, len(states))
            return batch_priors(out["policy"], states), self._values(out)
        return result

    def evaluate(self, states):
        return self.submit(states)()


class NetEvaluator(TorchEvaluator):
    """Evaluator for GoNet models."""

    def __init__(self, model, **kw):
        super().__init__(model, model.cfg.board_size, **kw)

    def features(self, states):
        return np.stack([make_features(s) for s in states])

    def _outputs(self, o):
        return o

    def _values(self, out):
        return np.tanh(out["value_logit"] / 2.0)  # = 2*sigmoid(logit) - 1


class LegacyEvaluator(TorchEvaluator):
    """Evaluator for the original project's AZNet checkpoints.

    Rebuilds the old 17-plane absolute-color history encoding from the state chain.
    The legal mask comes from the new engine (superko), so the old net can never
    play an illegal move. No MIN_PRIOR floor: priors are the plain masked softmax.
    """

    def __init__(self, model, **kw):
        super().__init__(model, model.board_size, **kw)

    def features(self, states):
        n = self.n
        x = np.zeros((len(states), 17, n, n), dtype=np.uint8)
        for i, s in enumerate(states):
            boards = []
            t = s
            while t is not None and len(boards) < 8:
                boards.append(t.board)
                t = t.prev
            boards.reverse()  # oldest first
            offset = 8 - len(boards)  # older slots stay zero (empty board padding)
            for j, b in enumerate(boards):
                b2 = b.reshape(n, n)
                x[i, 2 * (offset + j)] = b2 == 1
                x[i, 2 * (offset + j) + 1] = b2 == -1
            if s.to_play == 1:
                x[i, 16] = 1
        return x

    def _outputs(self, o):
        p, v = o
        return {"policy": p, "value": v}

    def _values(self, out):
        return out["value"].astype(np.float64)


class UniformEvaluator:
    """No network: uniform priors, value 0. Useful for testing MCTS in isolation."""

    num_evals = 0

    def evaluate(self, states):
        priors = []
        for s in states:
            moves = np.flatnonzero(s.legal_mask())
            priors.append((moves, np.full(len(moves), 1.0 / len(moves), dtype=np.float32)))
        self.num_evals += len(states)
        return priors, np.zeros(len(states))

    def submit(self, states):
        res = self.evaluate(states)
        return lambda: res


class RolloutEvaluator(UniformEvaluator):
    """Uniform priors, value from random playouts. A crude but unbiased reference player."""

    def __init__(self, rollouts=1, seed=None):
        self.rng = np.random.default_rng(seed)
        self.rollouts = rollouts

    def evaluate(self, states):
        priors, _ = super().evaluate(states)
        values = np.zeros(len(states))
        for i, s in enumerate(states):
            tot = 0.0
            for _ in range(self.rollouts):
                t = s
                while not t.is_terminal():
                    legal = np.flatnonzero(t.legal_mask()[:-1])
                    # don't fill own single-point eyes, otherwise random play never ends sensibly
                    legal = [m for m in legal if not _is_own_eye(t, m)]
                    t = t.play(int(self.rng.choice(legal)) if legal else t.rules.N)
                tot += 1.0 if t.winner() == s.to_play else -1.0
            values[i] = tot / self.rollouts
        return priors, values


def _is_own_eye(state, m):
    nb = state.rules.nb[m]
    for q in nb:
        if q < 0:
            break
        if state.board[q] != state.to_play:
            return False
    return True
