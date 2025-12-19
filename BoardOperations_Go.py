import numpy as np


def transform_board(board, index):  # CORRECT
    arr = np.array(board)

    is_stack = arr.ndim == 3
    if not is_stack:
        arr = arr[np.newaxis, ...]  # shape (1,9,9)

    flip = index % 2
    rotate = index // 2

    if flip:
        arr = np.flip(arr, axis=2)  # flip horizontally with axis 1 or 0?
    arr = np.rot90(arr, k=(rotate), axes=(1,2))

    if not is_stack:
        return arr.squeeze(0)
    return arr

def symmetries_of_board(board): # CORRECT
    return [transform_board(board, i) for i in range(8)]

    


def invert_sym_pi(pi, index, size): # CORRECT
    board_shape = (size, size)
    pi_board = pi[:-1].reshape(board_shape)
    is_stack = pi_board.ndim == 3
    if not is_stack:
        pi_board = pi_board[np.newaxis, ...]  # shape (1,9,9)

    rot = index // 2
    do_flip = index % 2
    arr = pi_board
    if do_flip:
        arr = np.flip(arr, axis=2)  # flip horizontally with axis 1 or 0?
    arr = np.rot90(arr, k=(rot), axes=(1,2))

    if not is_stack:
        arr = arr.squeeze(0)

    inv = arr.flatten()
    inv_pi = np.concatenate([inv, [pi[-1]]])
    return inv_pi

def inversions_of_pi(pi, size): # CORRECT
    return [invert_sym_pi(pi, i, size) for i in range(8)]



def make_input_planes(history_boards, to_play, size):

    planes = np.zeros((17, size, size), dtype=np.bool_)
    last_k = 8
    padded = [np.zeros((size,size), dtype=np.int8)]*(last_k - len(history_boards)) + history_boards[-last_k:]
    plane_idx = 0
    for b in padded:
        planes[plane_idx] = (b == 1).astype(np.bool_)  # black
        planes[plane_idx+1] = (b == -1).astype(np.bool_)  # white
        plane_idx += 2
    # to_play plane
    planes[-1] = np.full((size,size), 1.0 if to_play == 1 else 0.0, dtype=np.bool_)

    return planes  # shape (17,9,9)

