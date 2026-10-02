"""The 8 dihedral symmetries of the square board.

Symmetry k: optional transpose (k & 4), then flip rows (k & 1), then flip cols (k & 2).
Each transform has a precomputed index permutation, so boards, feature planes,
policies and ownership maps all transform the same way.
"""
import numpy as np

_PERM_CACHE = {}


def _transform_2d(a, k):
    if k & 4:
        a = np.swapaxes(a, -1, -2)
    if k & 1:
        a = np.flip(a, axis=-2)
    if k & 2:
        a = np.flip(a, axis=-1)
    return a


def permutations(n):
    """perm[k] is an index array: transformed_flat = flat[perm[k]] (excluding pass)."""
    if n not in _PERM_CACHE:
        idx = np.arange(n * n).reshape(n, n)
        _PERM_CACHE[n] = np.stack([_transform_2d(idx, k).reshape(-1) for k in range(8)])
    return _PERM_CACHE[n]


def inverse(k):
    # Transpose-then-flip compositions: all are involutions except the two 90-degree rotations.
    return {5: 6, 6: 5}.get(k, k)


def transform_planes(planes, k):
    """planes: (..., n, n) -> same shape under symmetry k."""
    return np.ascontiguousarray(_transform_2d(planes, k))


def transform_policy(pi, k, n):
    """pi: (..., n*n + 1) including pass at the end."""
    perm = permutations(n)[k]
    out = np.empty_like(pi)
    out[..., :n * n] = pi[..., perm]
    out[..., n * n] = pi[..., n * n]
    return out


def transform_flat(v, k, n):
    """v: (..., n*n) flat board values."""
    return v[..., permutations(n)[k]]


def invariant_symmetries(board, n):
    """Symmetries k (0 always included) that leave a flat board unchanged. They form a group."""
    perms = permutations(n)
    return [k for k in range(8) if np.array_equal(board[perms[k]], board)]
