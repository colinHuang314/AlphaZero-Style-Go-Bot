"""PUCT Monte Carlo Tree Search, batched across many independent trees.

Each call to `search` advances a list of roots (one per game) together: every
iteration descends once in each tree, then evaluates all the new leaves in a
single network batch. With ~100+ concurrent games this keeps the GPU busy
without virtual loss, so every tree gets exactly the AlphaZero search.

Value convention: node.Wc[i] is the summed value of child i from the point of
view of the player to move at `node` (the player choosing among the children).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..go.symmetry import permutations


@dataclass
class MCTSConfig:
    c_puct: float = 1.25
    fpu_reduction: float = 0.2       # unvisited child Q = parent Q - fpu_reduction * sqrt(visited prior mass)
    root_fpu_reduction: float = 0.0  # AlphaZero-like optimism at the root (helps noise explore)
    dirichlet_alpha: float = 0.15    # ~10 / typical number of legal moves on 9x9
    dirichlet_eps: float = 0.25
    symmetry_prune: bool = True      # in symmetric positions search one move per symmetry class


@njit(cache=True)
def _symmetry_mask(board, perms):
    """Bit k set (k = 1..7) if symmetry k leaves the flat board unchanged. Exits early on asymmetric boards."""
    mask = 0
    for k in range(1, perms.shape[0]):
        same = True
        for i in range(board.shape[0]):
            if board[perms[k, i]] != board[i]:
                same = False
                break
        if same:
            mask |= 1 << k
    return mask


def symmetry_prune(state, moves, priors):
    """Keep one move per symmetry class of a symmetric position, with the class's summed prior.

    Moves that map onto each other under a symmetry of the board lead to equivalent positions, so
    searching all of them splits the visits for nothing (an empty 9x9 board has 81 moves but only 15
    different ones). The previous board must have the symmetry too, so a ko can't make two moves differ.
    The kept move is the one nearest the top-right corner (smallest row - col, then smallest row): the
    bot opens in the upper right, as people do. Returns (moves, priors, orbits); orbits maps each kept
    index to the array of its class's legal moves (None when nothing was pruned).
    """
    r = state.rules
    perms = permutations(r.n)
    mask = _symmetry_mask(state.board, perms)
    if mask and state.prev is not None:
        mask &= _symmetry_mask(state.prev.board, perms)
    if not mask:
        return moves, priors, None
    ks = [0] + [k for k in range(1, 8) if mask >> k & 1]
    legal = {int(m): i for i, m in enumerate(moves)}
    keep, kept_p, orbits, done = [], [], {}, set()
    for i, m in enumerate(moves):
        m = int(m)
        if m in done:
            continue
        if m == r.N:                      # pass: its own class
            members = [m]
        else:
            members = sorted({int(perms[k][m]) for k in ks if int(perms[k][m]) in legal})
        done.update(members)
        rep = min(members, key=lambda q: (q // r.n - q % r.n, q // r.n)) if m != r.N else m
        orbits[len(keep)] = np.array(members, dtype=moves.dtype)
        keep.append(rep)
        kept_p.append(sum(float(priors[legal[q]]) for q in members))
    order = np.argsort(keep, kind="stable")   # keep moves sorted like the legal-move lists
    moves2 = np.array(keep, dtype=moves.dtype)[order]
    priors2 = np.array(kept_p, dtype=np.float32)[order]
    orbits2 = {j: orbits[int(o)] for j, o in enumerate(order)}
    return moves2, priors2, orbits2


class Node:
    __slots__ = ("state", "moves", "P", "P_raw", "Nc", "Wc", "N", "children",
                 "v_net", "expanded", "terminal", "terminal_value", "orbits")

    def __init__(self, state):
        self.state = state
        self.expanded = False
        self.N = 0
        self.orbits = None
        self.terminal = state.is_terminal()
        if self.terminal:
            s = state.score()  # a draw (possible only with integer komi, e.g. CGOS 9x9's 7.0) is worth 0
            self.terminal_value = 0.0 if s == 0 else (1.0 if (s > 0) == (state.to_play == 1) else -1.0)

    def expand(self, moves, priors, value, prune=False):
        if prune:
            moves, priors, self.orbits = symmetry_prune(self.state, moves, priors)
        self.moves = moves
        self.P = priors
        self.P_raw = priors
        self.Nc = np.zeros(len(moves), dtype=np.float64)
        self.Wc = np.zeros(len(moves), dtype=np.float64)
        self.children = [None] * len(moves)
        self.v_net = float(value)
        self.expanded = True

    def q(self):
        """Mean value of this node from its own side-to-move perspective."""
        return (self.v_net + self.Wc.sum()) / (1.0 + self.N)

    def child(self, idx):
        c = self.children[idx]
        if c is None:
            c = Node(self.state.play(int(self.moves[idx])))
            self.children[idx] = c
        return c

    def child_by_move(self, move):
        hits = np.flatnonzero(self.moves == move)
        if len(hits) == 0:
            raise ValueError(f"move {move} not legal at this node")
        return self.child(int(hits[0]))

    def child_or_new(self, move):
        """The subtree after `move` if the tree has it, else a fresh node (unexpanded node, or a move
        that symmetry pruning folded into an equivalent one: that subtree is in another orientation)."""
        if self.expanded and np.any(self.moves == move):
            return self.child_by_move(move)
        return Node(self.state.play(int(move)))

    def visit_distribution(self, n_actions):
        """Visit shares per action; a pruned symmetry class's visits are split evenly over its moves,
        so policy targets stay symmetric."""
        pi = np.zeros(n_actions, dtype=np.float32)
        if self.expanded and self.N > 0:
            share = self.Nc / self.Nc.sum()
            if self.orbits is None:
                pi[self.moves] = share
            else:
                for i, members in self.orbits.items():
                    pi[members] = share[i] / len(members)
        return pi


@njit(cache=True)
def _puct_argmax(P, Nc, Wc, N, v_net, c_puct, fpu_red):
    # parent Q (own perspective) and visited prior mass for first-play urgency
    wsum = 0.0
    pvis = 0.0
    for i in range(len(P)):
        if Nc[i] > 0:
            wsum += Wc[i]
            pvis += P[i]
    fpu = (v_net + wsum) / (1.0 + N) - fpu_red * np.sqrt(pvis)
    sq = np.sqrt(N)
    best = -1e30
    bi = 0
    for i in range(len(P)):
        q = Wc[i] / Nc[i] if Nc[i] > 0 else fpu
        u = q + c_puct * P[i] * sq / (1.0 + Nc[i])
        if u > best:
            best = u
            bi = i
    return bi


def select_child(node, cfg, is_root):
    if node.N == 0:
        return int(np.argmax(node.P))
    fpu_red = cfg.root_fpu_reduction if is_root else cfg.fpu_reduction
    return _puct_argmax(node.P, node.Nc, node.Wc, float(node.N), node.v_net, cfg.c_puct, fpu_red)


def backpropagate(path, value):
    """value: leaf value from the leaf's side-to-move perspective."""
    v = value
    for node, idx in reversed(path):
        v = -v
        node.Nc[idx] += 1.0
        node.Wc[idx] += v
        node.N += 1


class BatchedMCTS:
    def __init__(self, cfg: MCTSConfig, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def add_noise(self, root):
        cfg = self.cfg
        if cfg.dirichlet_eps <= 0 or len(root.moves) < 2:
            return
        noise = self.rng.dirichlet([cfg.dirichlet_alpha] * len(root.moves))
        root.P = ((1 - cfg.dirichlet_eps) * root.P_raw + cfg.dirichlet_eps * noise).astype(np.float32)

    def search(self, roots, evaluator, visits, noise=None):
        """Run search until every root has `visits[i]` total visits.

        roots:     list of Node (may already contain visits from tree reuse)
        evaluator: one evaluator for all roots, or a list with one per root
                   (e.g. arena games where each side uses a different network)
        visits:    int or list of target visit counts per root
        noise:     None or list of bools, whether to add Dirichlet noise at that root
        Returns number of network evaluations performed.

        Roots are split into two halves that alternate: while the GPU evaluates
        one half's leaves, the CPU descends the other half's trees.
        """
        n = len(roots)
        evs = list(evaluator) if isinstance(evaluator, (list, tuple)) else [evaluator] * n
        if np.isscalar(visits):
            visits = [int(visits)] * n
        evals = 0
        fresh = [i for i, r in enumerate(roots) if not r.expanded and not r.terminal]
        for job in self._submit([roots[i] for i in fresh], [None] * len(fresh), [evs[i] for i in fresh]):
            evals += self._complete(job)
        for i, r in enumerate(roots):
            if r.expanded:
                r.P = r.P_raw
                if noise is not None and noise[i]:
                    self.add_noise(r)

        active = [i for i, r in enumerate(roots) if not r.terminal and r.N < visits[i]]
        halves = [active[0::2], active[1::2]]
        pending = []
        while True:
            for h in (0, 1):
                halves[h] = [i for i in halves[h] if roots[i].N < visits[i]]
                jobs = self._descend(roots, halves[h], evs)
                for job in pending:
                    evals += self._complete(job)
                pending = jobs
            if not pending and not halves[0] and not halves[1]:
                break
        return evals

    def _descend(self, roots, idxs, evs):
        cfg = self.cfg
        leaves, paths, leaf_evs = [], [], []
        for i in idxs:
            node = roots[i]
            path = []
            is_root = True
            while node.expanded and not node.terminal:
                idx = select_child(node, cfg, is_root)
                path.append((node, idx))
                node = node.child(idx)
                is_root = False
            if node.terminal:
                backpropagate(path, node.terminal_value)
            else:
                leaves.append(node)
                paths.append(path)
                leaf_evs.append(evs[i])
        return self._submit(leaves, paths, leaf_evs)

    @staticmethod
    def _submit(leaves, paths, leaf_evs):
        """Group leaves by evaluator and launch one batch per evaluator."""
        groups = {}
        for leaf, path, ev in zip(leaves, paths, leaf_evs):
            groups.setdefault(id(ev), (ev, [], []))
            groups[id(ev)][1].append(leaf)
            groups[id(ev)][2].append(path)
        return [(ls, ps, ev.submit([l.state for l in ls])) for ev, ls, ps in groups.values()]

    def _complete(self, job):
        leaves, paths, result = job
        priors, values = result()
        for leaf, (moves, p), v in zip(leaves, priors, values):
            leaf.expand(moves, p, v, prune=self.cfg.symmetry_prune)
        for path, v in zip(paths, values):
            if path is not None:
                backpropagate(path, v)
        return len(leaves)


def _apply_virtual_loss(path, vl):
    for node, idx in path:
        node.Nc[idx] += vl
        node.Wc[idx] -= vl  # pretend the edge lost, steering the next descent elsewhere
        node.N += vl


def search_single(root, evaluator, cfg, target_visits, leaf_batch=16, virtual_loss=1.0, should_stop=None):
    """Search one tree, evaluating up to `leaf_batch` leaves per network call.

    Used by the interactive UI, where there is only one board. Virtual loss keeps
    the leaves in a batch distinct; it is removed before the real backup, so the
    final statistics are ordinary visit counts. Returns number of evaluations.
    """
    evals = 0
    if root.terminal:
        return 0
    if not root.expanded:
        priors, values = evaluator.evaluate([root.state])
        root.expand(priors[0][0], priors[0][1], values[0], prune=cfg.symmetry_prune)
        evals += 1
    while root.N < target_visits and not (should_stop and should_stop()):
        leaves, paths, seen = [], [], set()
        budget = min(leaf_batch, int(target_visits - root.N))
        for _ in range(budget):
            node, path, is_root = root, [], True
            while node.expanded and not node.terminal:
                idx = select_child(node, cfg, is_root)
                path.append((node, idx))
                node = node.child(idx)
                is_root = False
            if node.terminal:
                backpropagate(path, node.terminal_value)
                continue
            if id(node) in seen:
                continue  # collision: already being evaluated this batch
            seen.add(id(node))
            _apply_virtual_loss(path, virtual_loss)
            leaves.append(node)
            paths.append(path)
        if not leaves:
            continue
        priors, values = evaluator.evaluate([l.state for l in leaves])
        evals += len(leaves)
        for leaf, path, (moves, p), v in zip(leaves, paths, priors, values):
            _apply_virtual_loss(path, -virtual_loss)
            leaf.expand(moves, p, v, prune=cfg.symmetry_prune)
            backpropagate(path, v)
    return evals


def principal_variation(node, first_idx=None, max_len=12):
    """Most-visited line starting from child `first_idx` of `node`."""
    pv = []
    idx = first_idx
    while node is not None and node.expanded and len(pv) < max_len:
        if idx is None:
            if node.N == 0:
                break
            idx = int(np.argmax(node.Nc))
        if node.Nc[idx] < 1:
            break
        pv.append(int(node.moves[idx]))
        node = node.children[idx]
        idx = None
    return pv


def pick_move(root, temperature, rng):
    """Choose a move from root visit counts. temperature 0 = most visited (ties -> higher Q)."""
    if temperature <= 1e-6:
        best = np.flatnonzero(root.Nc == root.Nc.max())
        if len(best) > 1:
            q = root.Wc[best] / np.maximum(root.Nc[best], 1)
            best = best[q == q.max()]
        return int(root.moves[rng.choice(best)])
    w = root.Nc ** (1.0 / temperature)
    w /= w.sum()
    return int(root.moves[rng.choice(len(w), p=w)])
