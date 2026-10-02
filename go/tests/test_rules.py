import numpy as np
import pytest

from gozero.go import engine
from gozero.go.state import GoState, str_to_move


def setup(n, black=(), white=(), to_play=1):
    """Build a state from lists of coordinates like 'C3'."""
    s = GoState.new(n)
    b = s.board.copy()
    for m in black:
        b[str_to_move(m, n)] = 1
    for m in white:
        b[str_to_move(m, n)] = -1
    s = GoState(s.rules, b, to_play)
    return s


def mv(s, m):
    return str_to_move(m, s.rules.n)


def test_single_capture():
    s = setup(5, black=["B3", "D3", "C4"], white=["C3"])
    s2 = s.play(mv(s, "C2"))
    assert s2.board[mv(s, "C3")] == 0


def test_group_capture_in_corner():
    s = setup(5, black=["C1", "C2", "A3", "B3"], white=["A1", "B1", "A2"])
    s2 = s.play(mv(s, "B2"))
    for m in ["A1", "B1", "A2"]:
        assert s2.board[mv(s, m)] == 0
    assert s2.board[mv(s, "B2")] == 1


def test_suicide_illegal():
    s = setup(5, black=["A2", "B1"])
    s = GoState(s.rules, s.board, to_play=-1)
    assert s.legal_mask()[mv(s, "A1")] == 0


def test_multi_stone_suicide_illegal():
    # white A1, A2 with only liberty A3; black surrounds -> white playing A3 is suicide
    s = setup(5, black=["B1", "B2", "B3", "A4"], white=["A1", "A2"], to_play=-1)
    assert s.legal_mask()[mv(s, "A3")] == 0


def test_capture_beats_suicide():
    # A1 looks like suicide for black but captures white A2
    s = setup(5, black=["A3", "B2"], white=["A2", "B1"])
    assert s.legal_mask()[mv(s, "A1")] == 1
    s2 = s.play(mv(s, "A1"))
    assert s2.board[mv(s, "A2")] == 0 and s2.board[mv(s, "A1")] == 1


def ko_position():
    #   . X O .
    #   X O . O      black captures at C4 area; classic ko shape
    #   . X O .
    s = setup(5, black=["B5", "A4", "B3"], white=["C5", "B4", "D4", "C3"])
    return s


def test_ko_immediate_recapture_illegal():
    s = ko_position()
    s1 = s.play(mv(s, "C4"))           # black captures B4
    assert s1.board[mv(s, "B4")] == 0
    assert s1.legal_mask()[mv(s, "B4")] == 0  # white may not retake immediately
    s2 = s1.play(mv(s, "E1"))          # white ko threat elsewhere
    s3 = s2.play(mv(s, "E2"))          # black answers
    assert s3.legal_mask()[mv(s, "B4")] == 1  # now retake is legal


def test_positional_superko_bans_older_positions():
    # Any move recreating ANY earlier position is illegal, not just the previous one.
    s = setup(5, black=["A2"])
    target = s.play(mv(s, "C3"))  # position we pretend occurred long ago
    s.hist[s.nhist] = target.hash
    s.nhist += 1
    s._legal = None
    assert s.legal_mask()[mv(s, "C3")] == 0
    assert s.legal_mask()[mv(s, "C4")] == 1


def test_pass_rules_and_terminal():
    s = GoState.new(5)
    s = s.play(s.rules.N)
    assert not s.is_terminal() and s.passes == 1
    s = s.play(mv(s, "C3"))
    assert s.passes == 0
    s = s.play(s.rules.N).play(s.rules.N)
    assert s.is_terminal()


def test_scoring():
    s = GoState.new(5)
    assert s.score() == -3.5  # empty board: nobody owns anything, komi to white
    s = s.play(mv(s, "C3"))
    assert s.score() == 25 - 3.5  # one black stone owns the whole board
    s = s.play(mv(s, "C2"))
    assert s.score() == -3.5  # all empty points touch both colors


def test_incremental_hash_matches_recompute():
    rng = np.random.default_rng(0)
    for n in (5, 9):
        s = GoState.new(n)
        for _ in range(150):
            legal = np.flatnonzero(s.legal_mask())
            s = s.play(int(rng.choice(legal)))
            assert s.hash == engine.board_hash(s.board, s.rules.zob)
            if s.is_terminal():
                break


def test_no_repeated_positions_in_random_games():
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = GoState.new(5)
        while not s.is_terminal():
            legal = np.flatnonzero(s.legal_mask()[:-1])
            if len(legal) == 0 or rng.random() < 0.02:
                s = s.play(s.rules.N)
            else:
                s = s.play(int(rng.choice(legal)))
        h = s.hist[:s.nhist]
        assert len(set(h.tolist())) == len(h)
