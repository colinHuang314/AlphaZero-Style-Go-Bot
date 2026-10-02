"""Neural network input features.

All planes are relative to the player to move ("own" vs "opp"), so the same
network weights serve both colors. Every plane is spatially equivariant, so
symmetry augmentation can transform stored feature planes directly.

Planes (NUM_PLANES = 16):
  0      own stones
  1      opponent stones
  2-4    own stones in groups with 1 / 2 / >=3 liberties
  5-7    opponent stones in groups with 1 / 2 / >=3 liberties
  8      empty points that are illegal for the player to move (suicide / superko)
  9-12   location of the last 4 moves (9 = most recent), blank for passes
  13     1 everywhere if the previous move was a pass (next pass ends the game)
  14     1 everywhere if the player to move is black (komi asymmetry)
  15     1 everywhere (lets zero-padded convolutions detect the board edge)
"""
import numpy as np
from numba import njit

NUM_PLANES = 16


@njit(cache=True)
def _fill(board, to_play, gid, glibs, legal, last_moves, prev_pass, out):
    N = board.shape[0]
    for p in range(N):
        v = board[p]
        if v != 0:
            base = 0 if v == to_play else 1
            out[base, p] = 1
            libs = glibs[gid[p]]
            k = 2 if libs >= 3 else libs - 1  # 0,1,2 -> planes +0,+1,+2
            out[2 + 3 * base + k, p] = 1
        elif legal[p] == 0:
            out[8, p] = 1
    for i in range(4):
        m = last_moves[i]
        if 0 <= m < N:
            out[9 + i, m] = 1
    for p in range(N):
        if prev_pass:
            out[13, p] = 1
        if to_play == 1:
            out[14, p] = 1
        out[15, p] = 1


def make_features(state):
    """Return uint8 array (NUM_PLANES, n, n)."""
    r = state.rules
    legal = state.legal_mask()
    gid, glibs = state.groups()
    out = np.zeros((NUM_PLANES, r.N), dtype=np.uint8)
    lm = np.array(state.last_moves, dtype=np.int32)
    _fill(state.board, state.to_play, gid, glibs, legal, lm,
          state.last_moves[0] == r.N, out)
    return out.reshape(NUM_PLANES, r.n, r.n)
