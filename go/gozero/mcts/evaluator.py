"""Batched neural-network evaluators used by MCTS.

An evaluator turns a list of GoStates into (priors over legal moves, value) pairs.
Values are always from the perspective of the side to move, in [-1, 1].

`submit(states)` launches the GPU work and returns a callable that blocks for the
result, so MCTS can do CPU work for other games while the GPU is busy.
"""
import math

import numpy as np
import torch

from ..go import symmetry
from ..go.features import make_features

TRAIN_KOMI = 7.5  # the only komi the network has seen (komi is not a network input)


class ScoreUtility:
    """Turns the net's (win value, score) into the value the search maximizes, for play at a real komi.

    komi_fix: at an integer komi half a point from the training komi (CGOS 9x9 uses 7.0), the
      outcome "Black ahead by exactly komi on the board" is a draw, but the net counts it as a
      half-point win (for White at 7.0). The net's score estimate s (side to move, at 7.5) is taken
      as Normal(s, sigma); the chance that the game ends in that draw bin (width 2: board margins
      have one parity) moves that much win probability to a draw (value 0) instead of a win.
    score_weight: a small bonus for a bigger final margin at the real komi,
      score_weight * 2/pi * atan(margin / score_scale), as in KataGo's score utility. When the
      win is (nearly) certain it makes the search prefer the bigger win; when the loss is certain,
      the smaller loss (instead of passing while the opponent captures everything).
    Only the GTP engine and matches use this; self-play values (and so training targets) never do.
    """

    def __init__(self, komi_fix=True, score_weight=0.0, score_scale=5.0, sigma=2.0, train_komi=TRAIN_KOMI):
        self.komi_fix = komi_fix
        self.score_weight = score_weight
        self.score_scale = score_scale
        self.sigma = sigma
        self.train_komi = train_komi

    def _bin_prob(self, center, mu):
        """P(margin in [center - 1, center + 1]) for margin ~ Normal(mu, sigma)."""
        z = lambda x: 0.5 * (1.0 + math.erf((x - mu) / (self.sigma * math.sqrt(2.0))))
        return z(center + 1.0) - z(center - 1.0)

    def __call__(self, values, scores, states):
        out = np.asarray(values, dtype=np.float64).copy()
        for i, st in enumerate(states):
            komi = st.rules.komi
            d = komi - self.train_komi        # black's margin at the training komi that is a draw for real
            tp = st.to_play                   # +1 black, -1 white
            s = float(scores[i])              # net's margin, side to move, at the training komi
            v = out[i]
            if self.komi_fix and komi == int(komi) and abs(d) == 0.5:
                mu_black = s * tp
                p_draw = self._bin_prob(d, mu_black)
                # d < 0: those games are White wins at 7.5 (draws for real); d > 0: Black wins
                loser_of_mass = -1 if d < 0 else 1    # the color whose "win" becomes a draw
                p_that_win = (1.0 + v) / 2.0 if tp == loser_of_mass else (1.0 - v) / 2.0
                q = min(p_draw, p_that_win)
                v = v - q if tp == loser_of_mass else v + q
            if self.score_weight:
                margin = s - d * tp                    # side to move, at the real komi
                v = v + self.score_weight * (2.0 / math.pi) * math.atan(margin / self.score_scale)
            out[i] = v
        return out


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

    GRAPH_SIZES = (8, 16, 32, 64, 128, 256)  # batch sizes with a captured CUDA graph (padded up to these)

    def __init__(self, model, n, device=None, random_symmetry=True, use_amp=True, seed=None, cuda_graphs=True):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.random_symmetry = random_symmetry
        self.use_amp = use_amp and self.device.type == "cuda"
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.num_evals = 0
        self.utility = None  # optional ScoreUtility applied to the values (needs a "score" output)
        # CUDA graphs: an eager forward of this small net costs ~8 ms of CPU time just launching ~100
        # kernels, far more than the GPU work at search batch sizes (profiled Oct 7: self-play used ~14%
        # of the GPU). A replayed graph launches them all at once. The graphs read the model's own
        # parameter and buffer storage, so in-place weight updates (training, load_state_dict) apply.
        self.cuda_graphs = cuda_graphs and self.device.type == "cuda"
        self._graphs = {}  # batch size -> (graph, static input, static outputs)

    def _graph(self, size, planes):
        g = self._graphs.get(size)
        if g is None:
            x = torch.zeros(size, planes, self.n, self.n, device=self.device)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                for _ in range(3):  # warm-up on a side stream, as the CUDA graph docs require
                    self._outputs(self.model(x))
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                out = self._outputs(self.model(x))
            g = self._graphs[size] = (graph, x, out)
        return g

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
        xt = torch.from_numpy(x)
        if self.device.type == "cuda":
            # pinned: a copy from pageable memory makes the CPU wait for everything already queued on the
            # GPU (e.g. the other half's batch) before it returns
            xt = xt.pin_memory()
        xt = xt.to(self.device, non_blocking=True).float()
        b = len(states)
        if self.cuda_graphs and not self.model.training and b <= self.GRAPH_SIZES[-1]:
            size = next(s for s in self.GRAPH_SIZES if s >= b)
            graph, static_x, static_out = self._graph(size, xt.shape[1])
            static_x[:b].copy_(xt)
            static_x[b:].zero_()
            graph.replay()
            # copy out now: the next replay of this graph (e.g. the other half of the games, submitted
            # before this result is read) overwrites the static outputs; stream order keeps this safe
            return {k: v[:b].clone() for k, v in static_out.items()}
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
        done = None
        if self.device.type == "cuda":
            # Queue the copy back into pinned host memory now and wait later only for this batch's own
            # event. (A plain .cpu() in result() synchronizes the whole stream, so collecting one half of
            # the games also waited for the other half's batch queued behind it: no CPU/GPU overlap.)
            host = {}
            for key, v in gpu_out.items():
                host[key] = torch.empty(v.shape, dtype=torch.float32, pin_memory=True)
                host[key].copy_(v.float(), non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            gpu_out = host

        def result():
            if done is not None:
                done.synchronize()
            out = self._finish(gpu_out, k, len(states))
            values = self._values(out)
            if self.utility is not None and "score" in out:
                values = self.utility(values, out["score"], states)
            return batch_priors(out["policy"], states), values
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
