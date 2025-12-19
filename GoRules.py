import numpy as np
from collections import deque

class Game:
    def __init__(self):
        pass

    @classmethod
    def action_to_index(cls, action, board_size):
        if action is None:  # pass move
            return board_size * board_size
        r, c = action
        return r * board_size + c
    
    @classmethod
    def index_to_action(cls, action_idx, board_size):
        if action_idx == board_size * board_size:
            return None  # pass
        r = action_idx // board_size
        c = action_idx % board_size
        return (r, c)

    @classmethod
    def in_bounds(cls, x, y, board_size):
        return 0 <= x < board_size and 0 <= y < board_size
    
    @classmethod
    def neighbors(cls, x, y, board_size):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if cls.in_bounds(nx, ny, board_size):
                yield nx, ny
    
    @staticmethod
    def flood_group(board, x, y, board_size, visited=None):
        color = board[x, y]
        if color == 0:
            return []

        if visited is None:
            visited = np.zeros_like(board, dtype=bool)

        group = []
        stack = deque([(x, y)])

        while stack:
            a, b = stack.pop()
            if visited[a, b]:
                continue
            if board[a, b] != color:
                continue

            visited[a, b] = True
            group.append((a, b))

            # inline neighbors
            if a > 0 and not visited[a-1, b]:
                stack.append((a-1, b))
            if a < board_size-1 and not visited[a+1, b]:
                stack.append((a+1, b))
            if b > 0 and not visited[a, b-1]:
                stack.append((a, b-1))
            if b < board_size-1 and not visited[a, b+1]:
                stack.append((a, b+1))

        return group

    @classmethod
    def group_liberties(cls, board, group, board_size):
        libs = set()
        for (x,y) in group:
            for nx,ny in cls.neighbors(x,y, board_size):
                if board[nx,ny] == 0:
                    libs.add((nx,ny))
        return libs
    @classmethod
    def remove_group(cls, board, group):
        for x,y in group:
            board[x,y] = 0

    @classmethod
    def legal_moves(cls, board, to_play, prev_board, board_size):
        moves = []
        rows, cols = np.where(board == 0)
        for x, y in zip(rows, cols):
            if cls.is_legal(board, (x, y), to_play, prev_board, board_size):
                moves.append((x, y))

        moves.append(None)  # pass is always legal
        return moves

    @classmethod
    def get_legal_moves_mask(cls, board, to_play, prev_board, board_size):
        mask = np.zeros(board_size**2 + 1, dtype=np.float32)

        rows, cols = np.where(board == 0)
        for x, y in zip(rows, cols):
            idx = int(x) * board_size + int(y)
            if cls.is_legal(board, (int(x), int(y)), to_play, prev_board, board_size):
                mask[idx] = 1.0

        mask[-1] = 1.0  # pass is always legal
        return mask
    
    @classmethod
    def is_legal(cls, board, move, player, prev_board, board_size):
        if move is None:  # pass
            return True
        x, y = move
        if board[x, y] != 0:
            return False

        # Place the stone
        board[x, y] = player
        captured_stones = []

        # Check opponent groups for capture
        for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0 <= nx < board_size and 0 <= ny < board_size:
                if board[nx, ny] == -player:
                    g = cls.flood_group(board, nx, ny, board_size)
                    if not cls.group_has_liberty(board, g, board_size):
                        for gx, gy in g:
                            board[gx, gy] = 0
                            captured_stones.append((gx, gy))

        # Check own group liberties
        g2 = cls.flood_group(board, x, y, board_size)
        legal = cls.group_has_liberty(board, g2, board_size)

        # Ko rule
        if legal and prev_board is not None and np.array_equal(board, prev_board):
            legal = False

        # Undo move (restore board)
        board[x, y] = 0
        for gx, gy in captured_stones:
            board[gx, gy] = -player

        return legal

    @staticmethod
    def group_has_liberty(board, group, board_size):
        for (x, y) in group:
            if x > 0 and board[x-1, y] == 0: return True
            if x < board_size-1 and board[x+1, y] == 0: return True
            if y > 0 and board[x, y-1] == 0: return True
            if y < board_size-1 and board[x, y+1] == 0: return True
        return False

    @classmethod
    def apply_move(cls, board, move, player,board_size): #########################
        # returns new_board, captures_happened (bool)
        new = board.copy()
        if move is None or move[0] == board_size or move[1] == board_size:
            return new
        
        x,y = move
        new[x,y] = player
        for nx,ny in cls.neighbors(x,y,board_size):
            if new[nx,ny] == -player:
                g = cls.flood_group(new, nx, ny,board_size)
                if len(cls.group_liberties(new, g,board_size)) == 0:
                    cls.remove_group(new, g)
        # suicide should not happen if move is legal
        return new
    @classmethod
    def getKomi(cls, size):
        if size <= 5:
            return 3.5
        elif size == 7:
            return 5.5
        elif size == 9:
            return 7.5 # Chinese rules
        else:
            return 7.5
    @classmethod
    def final_score_tromp_taylor(cls, board, board_size): ##################
        komi=cls.getKomi(board_size)

        # Accept both numpy arrays and torch tensors
        if 'torch' in str(type(board)):
            board = board.detach().cpu().numpy()
        board = np.asarray(board, dtype=np.int8)

        H, W = board.shape
        visited = np.zeros((H, W), dtype=bool)

        # Stones on board
        black_stones = np.sum(board == 1)
        white_stones = np.sum(board == -1)

        black_terr = 0
        white_terr = 0

        territory_coords = []
        territory_values = []
        # 4 neighbor flood fill over empty points
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        for r in range(H):
            for c in range(W):
                if board[r, c] == 0 and not visited[r, c]:
                    stack = [(r, c)]
                    visited[r, c] = True
                    empties = 0
                    adj_colors = set()

                    temp_coords = []

                    while stack:
                        y, x = stack.pop()
                        empties += 1
                        temp_coords.append((x,y))
                        for dy, dx in dirs:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                v = board[ny, nx]
                                if v == 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))
                                elif v == 1:
                                    adj_colors.add(1)
                                elif v == -1:
                                    adj_colors.add(-1)

                    if adj_colors == {1}:
                        black_terr += empties
                        for coord in temp_coords:
                            territory_coords.append(coord)
                            territory_values.append(1)
                    elif adj_colors == {-1}:
                        white_terr += empties
                        for coord in temp_coords:
                            territory_coords.append(coord)
                            territory_values.append(-1)

        black_score = black_stones + black_terr
        white_score = white_stones + white_terr + komi
        return black_score - white_score, zip(territory_coords, territory_values)
    
