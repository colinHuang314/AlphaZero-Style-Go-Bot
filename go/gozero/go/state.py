"""Immutable-style Go game state built on the numba kernels in engine.py."""
import numpy as np

from . import engine

_CTX_CACHE = {}


class Rules:
    """Per-board-size constants shared by every state."""

    def __init__(self, n, komi=None, max_moves=None):
        self.n = n
        self.N = n * n
        self.komi = default_komi(n) if komi is None else komi
        self.max_moves = max_moves if max_moves is not None else 3 * n * n
        self.nb = engine.neighbor_table(n)
        self.zob = engine.zobrist_table(n)

    @classmethod
    def get(cls, n, komi=None, max_moves=None):
        key = (n, komi, max_moves)
        if key not in _CTX_CACHE:
            _CTX_CACHE[key] = cls(n, komi, max_moves)
        return _CTX_CACHE[key]


def default_komi(n):
    # Same komi values as the original project so results are comparable.
    if n <= 5:
        return 3.5
    if n == 7:
        return 5.5
    return 7.5


class GoState:
    """A position plus everything needed to continue the game.

    States are treated as immutable: play() returns a new state.
    """

    __slots__ = ("rules", "board", "to_play", "hash", "hist", "nhist", "passes",
                 "move_number", "last_moves", "prev", "_legal", "_gid", "_glibs")

    def __init__(self, rules, board=None, to_play=1):
        self.rules = rules
        self.board = np.zeros(rules.N, dtype=np.int8) if board is None else board
        self.to_play = to_play
        self.hash = engine.board_hash(self.board, rules.zob)
        self.hist = np.zeros(rules.max_moves + 2, dtype=np.uint64)
        self.hist[0] = self.hash
        self.nhist = 1
        self.passes = 0
        self.move_number = 0
        self.last_moves = (-1, -1, -1, -1)  # most recent first; N = pass, -1 = none
        self.prev = None  # previous state (used for board-history features of legacy nets)
        self._legal = None
        self._gid = None
        self._glibs = None

    @classmethod
    def new(cls, n, komi=None, max_moves=None):
        return cls(Rules.get(n, komi, max_moves))

    # ------------------------------------------------------------------ rules
    def _analyze(self):
        r = self.rules
        out = np.zeros(r.N + 1, dtype=np.uint8)
        gid, glibs = engine.legal_mask(self.board, self.to_play, r.nb, r.zob,
                                       self.hash, self.hist, self.nhist, out)
        self._legal, self._gid, self._glibs = out, gid, glibs

    def legal_mask(self):
        """uint8 array of length N+1; index N is pass."""
        if self._legal is None:
            self._analyze()
        return self._legal

    def groups(self):
        if self._gid is None:
            self._analyze()
        return self._gid, self._glibs

    def is_terminal(self):
        return self.passes >= 2 or self.move_number >= self.rules.max_moves

    def play(self, move):
        """Return the state after `move` (flat index, N = pass). Move must be legal."""
        r = self.rules
        s = GoState.__new__(GoState)
        s.rules = r
        s.to_play = -self.to_play
        s.move_number = self.move_number + 1
        s.last_moves = (move,) + self.last_moves[:3]
        s.prev = self
        s._legal = s._gid = s._glibs = None
        s.hist = self.hist.copy()
        if move == r.N:
            s.board = self.board  # boards are never mutated in place, safe to share
            s.hash = self.hash
            s.passes = self.passes + 1
            s.nhist = self.nhist
        else:
            s.board = self.board.copy()
            s.hash, _ = engine.play_move(s.board, move, self.to_play, r.nb, r.zob, self.hash)
            s.passes = 0
            s.hist[self.nhist] = s.hash
            s.nhist = self.nhist + 1
        return s

    # ---------------------------------------------------------------- scoring
    def ownership(self):
        """Tromp-Taylor ownership per point: +1 black, -1 white, 0 neutral."""
        return engine.area_ownership(self.board, self.rules.nb)

    def score(self):
        """Black area minus white area minus komi (positive = black wins)."""
        return float(self.ownership().sum()) - self.rules.komi

    def winner(self):
        return 1 if self.score() > 0 else -1

    # ---------------------------------------------------------------- helpers
    def to_2d(self):
        return self.board.reshape(self.rules.n, self.rules.n)

    def __repr__(self):
        n = self.rules.n
        chars = {0: ".", 1: "X", -1: "O"}
        rows = [" ".join(chars[int(v)] for v in self.board[r * n:(r + 1) * n]) for r in range(n)]
        side = "X" if self.to_play == 1 else "O"
        return "\n".join(rows) + f"\n(to play: {side}, move {self.move_number}, passes {self.passes})"


def move_to_str(move, n):
    if move == n * n:
        return "pass"
    if move < 0:
        return "-"
    cols = "ABCDEFGHJKLMNOPQRST"  # Go convention skips I
    r, c = divmod(move, n)
    return f"{cols[c]}{n - r}"


def str_to_move(s, n):
    s = s.strip().upper()
    if s == "PASS":
        return n * n
    cols = "ABCDEFGHJKLMNOPQRST"
    c = cols.index(s[0])
    r = n - int(s[1:])
    return r * n + c
