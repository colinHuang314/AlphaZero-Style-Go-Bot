"""Numba kernels for Go rules on a flat board.

Board layout: 1-D int8 array of length n*n, index p = row * n + col.
Values: 0 empty, 1 black, -1 white. The pass move is index n*n.

Rules implemented (Tromp-Taylor style):
  * captures of opponent groups with no liberties
  * suicide is illegal
  * positional superko: a move may not recreate any earlier board position
  * area scoring (stones + empty regions touching only one color), komi to white
"""
import numpy as np
from numba import njit

EMPTY, BLACK, WHITE = 0, 1, -1


def neighbor_table(n):
    """(n*n, 4) array of neighbor indices, -1 where off-board."""
    nb = -np.ones((n * n, 4), dtype=np.int32)
    for r in range(n):
        for c in range(n):
            p = r * n + c
            k = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    nb[p, k] = rr * n + cc
                    k += 1
    return nb


def zobrist_table(n, seed=12345):
    """(n*n, 2) random uint64 keys; column 0 = black stone, column 1 = white stone."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, 2**63 - 1, size=(n * n, 2), dtype=np.int64).astype(np.uint64)


@njit(cache=True)
def _zkey(zob, p, color):
    return zob[p, 0] if color == 1 else zob[p, 1]


@njit(cache=True)
def compute_groups(board, nb):
    """Label every stone with a group id.

    Returns (gid, glibs, ghash_dummy_size):
      gid[p]   : group id of the stone at p, or -1 for empty
      glibs[g] : number of distinct liberties of group g
      ngroups  : number of groups
    """
    N = board.shape[0]
    gid = -np.ones(N, dtype=np.int32)
    glibs = np.zeros(N, dtype=np.int32)
    stack = np.empty(N, dtype=np.int32)
    lib_stamp = -np.ones(N, dtype=np.int32)
    ng = 0
    for s in range(N):
        if board[s] == 0 or gid[s] >= 0:
            continue
        color = board[s]
        top = 0
        stack[top] = s
        top += 1
        gid[s] = ng
        libs = 0
        while top > 0:
            top -= 1
            p = stack[top]
            for k in range(4):
                q = nb[p, k]
                if q < 0:
                    break
                v = board[q]
                if v == 0:
                    if lib_stamp[q] != ng:
                        lib_stamp[q] = ng
                        libs += 1
                elif v == color and gid[q] < 0:
                    gid[q] = ng
                    stack[top] = q
                    top += 1
        glibs[ng] = libs
        ng += 1
    return gid, glibs, ng


@njit(cache=True)
def group_hashes(board, gid, ng, zob):
    """XOR of zobrist keys of all stones in each group."""
    gh = np.zeros(max(ng, 1), dtype=np.uint64)
    for p in range(board.shape[0]):
        g = gid[p]
        if g >= 0:
            gh[g] ^= _zkey(zob, p, board[p])
    return gh


@njit(cache=True)
def _in_history(h, hist, nhist):
    for i in range(nhist):
        if hist[i] == h:
            return True
    return False


@njit(cache=True)
def legal_mask(board, color, nb, zob, cur_hash, hist, nhist, out):
    """Fill out[0..N] with 1 for legal moves (out[N] = pass, always legal).

    hist[:nhist] are hashes of all earlier positions (for positional superko).
    Returns (gid, glibs) so callers can reuse the group analysis for features.
    """
    N = board.shape[0]
    gid, glibs, ng = compute_groups(board, nb)
    gh = group_hashes(board, gid, ng, zob)
    seen = np.empty(4, dtype=np.int32)
    for p in range(N):
        out[p] = 0
        if board[p] != 0:
            continue
        has_lib = False
        nseen = 0
        new_hash = cur_hash ^ _zkey(zob, p, color)
        for k in range(4):
            q = nb[p, k]
            if q < 0:
                break
            v = board[q]
            if v == 0:
                has_lib = True
            elif v == color:
                if glibs[gid[q]] > 1:
                    has_lib = True
            else:
                g = gid[q]
                if glibs[g] == 1:
                    dup = False
                    for i in range(nseen):
                        if seen[i] == g:
                            dup = True
                    if not dup:
                        seen[nseen] = g
                        nseen += 1
                        new_hash ^= gh[g]
                    has_lib = True
        if not has_lib:
            continue  # suicide
        if _in_history(new_hash, hist, nhist):
            continue  # positional superko
        out[p] = 1
    out[N] = 1
    return gid, glibs


@njit(cache=True)
def play_move(board, p, color, nb, zob, cur_hash):
    """Place a stone of `color` at p (assumed legal) in-place. Returns (new_hash, n_captured)."""
    N = board.shape[0]
    h = cur_hash
    if p == N:
        return h, 0
    board[p] = color
    h ^= _zkey(zob, p, color)
    captured = 0
    stack = np.empty(N, dtype=np.int32)
    visited = np.zeros(N, dtype=np.uint8)
    members = np.empty(N, dtype=np.int32)
    for k in range(4):
        q = nb[p, k]
        if q < 0:
            break
        if board[q] != -color or visited[q]:
            continue
        # flood-fill opponent group at q, check liberties
        top = 0
        nm = 0
        stack[top] = q
        top += 1
        visited[q] = 1
        has_lib = False
        while top > 0:
            top -= 1
            s = stack[top]
            members[nm] = s
            nm += 1
            for kk in range(4):
                t = nb[s, kk]
                if t < 0:
                    break
                v = board[t]
                if v == 0:
                    has_lib = True
                elif v == -color and not visited[t]:
                    visited[t] = 1
                    stack[top] = t
                    top += 1
        if not has_lib:
            for i in range(nm):
                s = members[i]
                board[s] = 0
                h ^= _zkey(zob, s, -color)
            captured += nm
    return h, captured


@njit(cache=True)
def board_hash(board, zob):
    h = np.uint64(0)
    for p in range(board.shape[0]):
        if board[p] != 0:
            h ^= _zkey(zob, p, board[p])
    return h


@njit(cache=True)
def area_ownership(board, nb):
    """Tromp-Taylor ownership: +1 black, -1 white, 0 neutral, per point."""
    N = board.shape[0]
    own = np.zeros(N, dtype=np.int8)
    visited = np.zeros(N, dtype=np.uint8)
    stack = np.empty(N, dtype=np.int32)
    region = np.empty(N, dtype=np.int32)
    for s in range(N):
        if board[s] != 0:
            own[s] = board[s]
            continue
        if visited[s]:
            continue
        top = 0
        nr = 0
        stack[top] = s
        top += 1
        visited[s] = 1
        touch_b = False
        touch_w = False
        while top > 0:
            top -= 1
            p = stack[top]
            region[nr] = p
            nr += 1
            for k in range(4):
                q = nb[p, k]
                if q < 0:
                    break
                v = board[q]
                if v == 0:
                    if not visited[q]:
                        visited[q] = 1
                        stack[top] = q
                        top += 1
                elif v == 1:
                    touch_b = True
                else:
                    touch_w = True
        val = 0
        if touch_b and not touch_w:
            val = 1
        elif touch_w and not touch_b:
            val = -1
        for i in range(nr):
            own[region[i]] = val
    return own
