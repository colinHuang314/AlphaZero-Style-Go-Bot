"""Life-and-death / tactics problem files (see problems/README.md).

One problem per file, plain text (`.txt`) or SGF (`.sgf`). Shared by
tools/diagnose.py (command-line evaluation) and the UI's Puzzles tab.
"""
import os
import re

import numpy as np

from ..go import engine
from ..go.state import GoState, move_to_str, str_to_move


def build(n, black, white, to_play):
    """Set up a position from coordinate lists like ['C3', 'D4']."""
    s = GoState.new(n)
    b = s.board.copy()
    for m in black:
        b[str_to_move(m, n)] = 1
    for m in white:
        b[str_to_move(m, n)] = -1
    return GoState(s.rules, b, to_play)


def zero_liberty_groups(board, n):
    """Stones in groups with no liberties (an impossible setup position)."""
    s = GoState.new(n)
    gid, glibs, _ = engine.compute_groups(board.astype(np.int8), s.rules.nb)
    return [p for p in range(n * n) if board[p] != 0 and glibs[gid[p]] == 0]


def _parse_txt(text):
    d = {}
    for line in text.splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, v = line.split(":", 1)
            d[k.strip().lower()] = v.strip()
    return d


def load_problem(path, n=9):
    """Returns dict(name, file, state, answers: set | None, inside: set | None, note)."""
    text = open(path, encoding="utf-8").read()
    base = os.path.basename(path)
    if path.endswith(".sgf"):
        cols = "abcdefghijklmnopqrs"
        m = re.search(r"SZ\[(\d+)\]", text)
        size = int(m.group(1)) if m else n
        pt = lambda s: cols.index(s[1]) * size + cols.index(s[0])
        blk = lambda tag: [pt(x) for g in re.findall(tag + r"((?:\[[a-s]{2}\])+)", text)
                           for x in re.findall(r"\[([a-s]{2})\]", g)]
        s = GoState.new(size)
        b = s.board.copy()
        for p in blk(r"\bAB"):
            b[p] = 1
        for p in blk(r"\bAW"):
            b[p] = -1
        pl = re.search(r"PL\[([BW])\]", text)
        moves = re.findall(r";\s*([BW])\[([a-s]{0,2})\]", text)
        first = (1 if moves[0][0] == "B" else -1) if moves else (1 if not pl or pl.group(1) == "B" else -1)
        s = GoState(s.rules, b, first)
        for _, mv in moves:
            s = s.play(size * size if mv in ("", "tt") else pt(mv))
        if pl and not moves:
            s = GoState(s.rules, s.board, 1 if pl.group(1) == "B" else -1)
        c = re.search(r"C\[((?:[^\]\\]|\\.)*)\]", text)
        d = _parse_txt(c.group(1) if c else "")
        d.setdefault("name", base)
    else:
        d = _parse_txt(text)
        size = n
        s = build(size, d.get("black", "").split(), d.get("white", "").split(),
                  1 if d.get("to_play", "B").upper().startswith("B") else -1)
    ans = d.get("answer", "").split()
    tenuki = [a.lower() for a in ans] == ["tenuki"]
    answers = None if tenuki else {str_to_move(a, size) for a in ans}
    inside = {str_to_move(a, size) for a in d.get("inside", "").split()} if tenuki else None
    if not ans or (tenuki and not inside):
        raise ValueError(f"{base}: needs 'answer:' (and 'inside:' when answer is tenuki)")
    for mv in (answers or set()):
        if not s.legal_mask()[mv]:
            raise ValueError(f"{base}: answer {move_to_str(mv, size)} is not a legal move")
    return {"name": d.get("name", base), "file": base, "state": s, "answers": answers,
            "inside": inside, "note": d.get("note", "")}


def is_correct(problem, move):
    if problem["answers"] is not None:
        return move in problem["answers"]
    return move not in problem["inside"]


def list_problems(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith((".txt", ".sgf")))


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:60] or "puzzle"


def save_problem(folder, filename, name, board, to_play, n, marks, tenuki, note=""):
    """Write a .txt problem. marks = correct moves, or (if tenuki) the points counted as wrong."""
    os.makedirs(folder, exist_ok=True)
    coords = lambda color: " ".join(move_to_str(p, n) for p in range(n * n) if board[p] == color)
    lines = [f"name: {name}", f"to_play: {'B' if to_play == 1 else 'W'}",
             f"black: {coords(1)}", f"white: {coords(-1)}"]
    marked = " ".join(move_to_str(p, n) for p in sorted(marks))
    if tenuki:
        lines += ["answer: tenuki", f"inside: {marked}"]
    else:
        lines.append(f"answer: {marked}")
    if note:
        lines.append(f"note: {' '.join(note.split())}")
    lines.append("source: go-zero UI")
    path = os.path.join(folder, filename)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    load_problem(path, n)  # round-trip check: raises if the file is not a valid problem
    return path
