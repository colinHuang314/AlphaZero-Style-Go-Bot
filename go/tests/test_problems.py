import os

import numpy as np

from gozero.eval.problems import load_problem, save_problem, zero_liberty_groups
from gozero.go.state import str_to_move


def test_txt_and_sgf_formats_agree(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("name: capture\nto_play: B\nblack: D4 E5 F4\nwhite: E4\nanswer: E3\n")
    sgf = tmp_path / "b.sgf"
    sgf.write_text("(;GM[1]SZ[9]KM[7.5]PL[B]AB[df][ee][ff]AW[ef]C[answer: E3\nnote: capture])")
    a, b = load_problem(str(txt)), load_problem(str(sgf))
    assert (a["state"].board == b["state"].board).all()
    assert a["state"].to_play == b["state"].to_play == 1
    assert a["answers"] == b["answers"] == {str_to_move("E3", 9)}


def test_sgf_with_moves_uses_final_position(tmp_path):
    sgf = tmp_path / "g.sgf"
    sgf.write_text("(;GM[1]SZ[9]C[answer: C3];B[ee];W[cc];B[gg])")
    p = load_problem(str(sgf))
    assert p["state"].to_play == -1 and p["state"].move_number == 3


def test_tenuki_requires_inside(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("to_play: B\nblack: A2 B2 B1\nwhite:\nanswer: tenuki\n")
    try:
        load_problem(str(f))
        assert False, "should require inside:"
    except ValueError:
        pass


def test_shipped_problems_load():
    folder = os.path.join(os.path.dirname(__file__), "..", "problems")
    for name in os.listdir(folder):
        if name.endswith((".txt", ".sgf")):
            load_problem(os.path.join(folder, name))


def test_save_roundtrip_and_tenuki(tmp_path):
    n = 9
    b = np.zeros(n * n, np.int8)
    for m in ["D1", "D2", "D3", "C3", "C4", "B4", "A4"]:
        b[str_to_move(m, n)] = 1
    for m in ["C1", "C2", "B2", "B3", "A3"]:
        b[str_to_move(m, n)] = -1
    save_problem(str(tmp_path), "bent.txt", "bent three", b, 1, n, {str_to_move("A1", n)}, tenuki=False, note="kill")
    p = load_problem(str(tmp_path / "bent.txt"))
    assert (p["state"].board == b).all() and p["answers"] == {str_to_move("A1", n)} and p["note"] == "kill"
    save_problem(str(tmp_path), "t.txt", "settled", b, 1, n, {str_to_move("A1", n), str_to_move("B1", n)}, tenuki=True)
    t = load_problem(str(tmp_path / "t.txt"))
    assert t["answers"] is None and t["inside"] == {str_to_move("A1", n), str_to_move("B1", n)}


def test_zero_liberty_detection():
    n = 9
    b = np.zeros(n * n, np.int8)
    b[str_to_move("A1", n)] = -1
    b[str_to_move("A2", n)] = 1
    b[str_to_move("B1", n)] = 1
    assert zero_liberty_groups(b, n) == [str_to_move("A1", n)]
