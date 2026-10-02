# Life-and-death problems

The easiest way to make problems is the **Puzzles** tab in the UI
(`python -m gozero.ui.server`): place stones, mark the answer, press Save.
It writes the text format below. Each file is one problem, and two formats are accepted.

## 1. Plain text (`.txt`), easiest to write by hand

```
name: bent three in the corner, black kills
to_play: B
black: D1 D2 D3 C3 C4 B4 A4
white: C1 C2 B2 B3 A3
answer: A1
note: vital point of bent three; if black plays elsewhere, white A1 makes two eyes
```

Coordinates are like the UI: columns A–J (no I), rows 1–9 from the bottom.
`answer` may list several equally correct moves (`answer: A1 B1`) or `pass`.
Use `answer: tenuki` when any move outside the group is correct (the group
is already settled); then `inside:` must list the points that count as wrong.

## 2. SGF (`.sgf`), from any editor (Sabaki, the go-zero UI, etc.)

Setup stones with `AB[..]` / `AW[..]`, played moves are also fine (e.g. a real
game exported from the UI; the position after the last move is used). The side
to move comes from `PL[B]` / `PL[W]`, or else from the move sequence. Put the
answer in the root comment:

```
(;GM[1]SZ[9]KM[7.5]PL[B]AB[df][ee][ff]AW[ef]C[answer: E3
note: capture in one (black D4 E5 F4, white E4)])
```

Check a problem set with:

```
python tools/diagnose.py runs/eval_night2/candidate_c162.pt --problems problems/
```
