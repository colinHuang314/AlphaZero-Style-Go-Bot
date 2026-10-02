from gozero.gtp import GTPEngine
from gozero.mcts.evaluator import UniformEvaluator


def engine(n=5, **kw):
    return GTPEngine(UniformEvaluator(), n=n, visits=kw.pop("visits", 32), **kw)


def ok(resp):
    assert resp.startswith("="), resp
    assert resp.endswith("\n\n")
    return resp[1:].strip()


def test_basic_commands_and_ids():
    e = engine()
    assert e.handle("7 protocol_version") == "=7 2\n\n"
    assert ok(e.handle("known_command genmove")) == "true"
    assert ok(e.handle("known_command frobnicate")) == "false"
    assert e.handle("  # just a comment") is None
    assert e.handle("3 frobnicate").startswith("?3 ")
    assert e.handle("boardsize 9").startswith("? unacceptable size")
    ok(e.handle("boardsize 5"))


def test_play_validation():
    e = engine()
    ok(e.handle("play b C3"))
    assert e.handle("play w C3").startswith("? illegal move")
    assert e.handle("play w F1").startswith("? invalid coordinate")   # off a 5x5 board
    assert e.handle("play w A6").startswith("? invalid coordinate")
    assert e.handle("play w I1").startswith("? invalid coordinate")   # GTP skips I
    ok(e.handle("play w pass"))
    assert e.state.move_number == 2


def test_genmove_is_legal_and_advances():
    e = engine()
    for color in ("b", "w", "b", "w"):
        mv = ok(e.handle(f"genmove {color}"))
        assert mv == "pass" or mv[0] in "abcde"
    assert e.state.move_number == 4
    assert e.state.to_play == 1


def test_out_of_turn_play_inserts_pass():
    e = engine()
    ok(e.handle("play b C3"))
    ok(e.handle("play b B2"))  # white skipped a turn
    assert e.state.move_number == 3
    assert e.state.to_play == -1


def test_undo_and_final_score():
    e = engine()
    ok(e.handle("komi 0.5"))
    ok(e.handle("play b C3"))
    assert ok(e.handle("final_score")) == "B+24.5"   # lone stone owns the whole 5x5 board
    ok(e.handle("undo"))
    assert ok(e.handle("final_score")) == "W+0.5"
    assert e.handle("undo").startswith("?")


def test_komi_mid_game_keeps_moves():
    e = engine()
    ok(e.handle("play b C3"))
    ok(e.handle("play w B2"))
    ok(e.handle("komi 3"))
    assert e.state.move_number == 2
    assert e.state.rules.komi == 3


def test_integer_komi_draw_is_worth_zero():
    from gozero.go.state import GoState
    from gozero.mcts.mcts import Node
    s = GoState.new(5, komi=25.0)
    s = s.play(12)          # black C3 owns all 25 points: score 25 - 25 = 0
    s = s.play(25).play(25)  # two passes end the game
    assert s.score() == 0
    assert Node(s).terminal_value == 0.0
    s7 = GoState.new(5, komi=24.5).play(12).play(25).play(25)
    assert Node(s7).terminal_value == (1.0 if s7.to_play == 1 else -1.0)


def test_symmetric_moves_share_a_class():
    import numpy as np
    from gozero.go.state import GoState
    from gozero.gtp import move_classes
    s = GoState.new(5)
    moves = np.flatnonzero(s.legal_mask())
    cls = move_classes(s, moves)
    # empty 5x5: corner, edge-next-to-corner, edge-middle, 2-2 point, 2-3 point, center, pass
    assert len(np.unique(cls)) == 7
    assert len({cls[i] for i, m in enumerate(moves) if m in (0, 4, 20, 24)}) == 1
    s1 = s.play(12)                      # black center: still fully symmetric
    m1 = np.flatnonzero(s1.legal_mask())
    assert len(np.unique(move_classes(s1, m1))) == 6
    s2 = s1.play(0)                      # white corner: only the mirror through that corner is left
    m2 = np.flatnonzero(s2.legal_mask())
    c2 = move_classes(s2, m2)
    pos = {int(m): i for i, m in enumerate(m2)}
    assert c2[pos[1]] == c2[pos[5]]      # the two points next to the white stone mirror each other
    assert c2[pos[4]] == c2[pos[20]]     # so do the two empty corners beside it
    assert c2[pos[4]] != c2[pos[24]]     # but not the opposite corner


def test_time_managed_genmove():
    e = engine(n=9, visits=64, time_manage=True)
    for color in ("b", "w", "b", "w", "b", "w"):
        mv = ok(e.handle(f"genmove {color}"))
        assert mv == "pass" or mv[0] in "abcdefghj"
    assert e.state.move_number == 6
    assert e.tree.N <= 64 * e.tm_ext + e.TM_CHUNK
    ok(e.handle("time_left b 0.1 0"))
    ok(e.handle("genmove b"))           # still answers with almost no time


def test_clock_derived_target():
    import re
    lines = []
    e = GTPEngine(UniformEvaluator(), n=9, visits=100000, time_manage=True, tm_use=0.5, log=lines.append)
    for color in ("b", "w", "b"):
        ok(e.handle(f"time_left {color} 120 0"))   # budget (90/60 - 0.3) = 1.2 s, half of it per normal move
        ok(e.handle(f"genmove {color}"))
    assert e.rate is not None and e.rate > 0
    moves = [l for l in lines if l.startswith("move ")]
    targets = [int(re.search(r"target (\d+)", l).group(1)) for l in moves]
    assert targets[0] == 100000 and targets[-1] < 100000   # once the speed is known, the clock sets the target
    assert all("reused)" in l for l in moves)
    assert any(l.startswith("new game") for l in lines) and any(l.startswith("time_left b 120") for l in lines)


def test_clock_budget():
    e = engine(n=9)
    assert e.move_budget(1) is None
    ok(e.handle("time_left b 300 0"))
    assert 4 < e.move_budget(1) < 6          # 300 s over ~60 remaining moves
    ok(e.handle("time_left b 100 0"))
    assert abs(e.move_budget(1) - (70 / 60 - 0.3)) < 1e-6   # the last 30 s are never planned for
    e.state.move_number = 150                 # long game: still spread over at least 20 more moves
    assert abs(e.move_budget(1) - (70 / 20 - 0.3)) < 1e-6
    e.state.move_number = 0
    ok(e.handle("time_left b 300 0"))
    ok(e.handle("time_left b 30 5"))
    assert abs(e.move_budget(1) - 5.7) < 1e-6  # byo-yomi: 30 s for 5 stones, minus latency
    ok(e.handle("time_left b 0.1 0"))
    ok(e.handle("genmove b"))                 # still answers with almost no time
