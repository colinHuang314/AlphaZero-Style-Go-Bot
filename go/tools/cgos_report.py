"""Report on CGOS games from the client log and the engine's --log-file (no GPU needed).

    python tools/cgos_report.py [--client runs/cgos/log/cgos.log] [--moves runs/cgos/log/moves.log]
                                [--blindspots runs/cgos/blindspots] [--drop 0.25]

Per game: opponent and rating, our color, result, clock used, visits / reuse, the win
estimate (Q of the move played, from our side) over the game, the biggest drops between
consecutive own moves (where the game turned: candidate positions to study or turn into
puzzles), and passes made while the opponent kept playing. Then totals by color and by
opponent rating band.

--blindspots DIR writes, for every drop of at least --drop between two of our moves, the
game cut off at two points, as SGFs the UI can open (Load SGF):
  *_before.sgf  just before our move (was our move the mistake?)
  *_after.sgf   after the opponent's reply (what did the bot not see coming?)
plus DIR/index.txt listing them. The DIR is rewritten on every run.
"""
import glob
import shutil
import argparse
import datetime
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ts(s):
    return datetime.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def client_games(path):
    games = []
    for l in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"Starting game against (.+?)\((\d+)\??\)\. Local engine .*?rated (\d+)\??\) is playing (black|white)", l)
        if m:
            games.append(dict(start=ts(l), opp=m.group(1), opp_rating=int(m.group(2)),
                              our_rating=int(m.group(3)), color=m.group(4)))
            continue
        m = re.search(r"Game over\. Result: (\S+)", l)
        if m and games and "result" not in games[-1]:
            games[-1]["result"] = m.group(1)
            games[-1]["end"] = ts(l)
        if "Local engine won" in l and games:
            games[-1]["won"] = True
        if "Local engine lost" in l and games:
            games[-1]["won"] = False
    for g in games:  # the client logs a draw as "lost"; count it as half a win
        if g.get("result", "").lower().startswith("draw") or g.get("result") == "0":
            g["won"] = 0.5
    return games


def engine_moves(path):
    out = []
    for l in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"(\S+ \S+) move (\d+): (\S+) visits (\d+) \((\d+) reused\) q ([-+][\d.]+) time ([\d.]+)s", l)
        if m:
            out.append(dict(t=ts(m.group(1)), num=int(m.group(2)), mv=m.group(3), visits=int(m.group(4)),
                            reused=int(m.group(5)), q=float(m.group(6)), time=float(m.group(7)),
                            ext="extended" in l))
    return out


def find_sgf(sgf_dir, end):
    """The client names each game record after the time it ended (to the second)."""
    for f in glob.glob(os.path.join(sgf_dir, "*.sgf")):
        m = re.match(r"(\d{4}-\d\d-\d\d-\d\d-\d\d-\d\d)-", os.path.basename(f))
        if m and abs((datetime.datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S") - end).total_seconds()) <= 2:
            return f
    return None


def cut_sgf(text, n_moves, comment):
    """The game record keeping only its first n_moves moves, with a comment on the root."""
    head = text[:text.index(";B[") if ";B[" in text else len(text)]
    head = head.rstrip().rstrip("(").rstrip()
    moves = re.findall(r";\s*([BW])\[([a-s]{0,2})\]", text)[:n_moves]
    body = "".join(f";{c}[{m}]" for c, m in moves)
    note = comment.replace("]", "\\]")
    return f"{head}C[{note}]\n{body})\n"


def write_blindspots(out_dir, games, moves, sgf_dir, threshold):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    index = [f"Positions where go-zero's win estimate (Q of its move, from its side) fell by >= {threshold:g}",
             "between two of its moves. Open in the UI with Load SGF; the last move shown is the one before the",
             "position. CGOS komi is 7.0 (the UI scores with 7.5), so evaluations differ slightly.", ""]
    count = 0
    for g in games:
        mv = [m for m in moves if g["start"] <= m["t"] <= g["end"]]
        f = find_sgf(sgf_dir, g["end"]) if mv else None
        if not f:
            continue
        text = open(f, encoding="utf-8", errors="replace").read()
        opp = re.sub(r"[^A-Za-z0-9_-]", "", g["opp"])
        for a, b in zip(mv, mv[1:]):
            if a["q"] - b["q"] < threshold:
                continue
            count += 1
            stem = f"{g['start']:%m%d-%H%M}_vs-{opp}_m{a['num']:03d}"
            desc = (f"{g['start']:%a %H:%M} as {g['color']} vs {g['opp']} ({g['opp_rating']}), result {g['result']}. "
                    f"Q {a['q']:+.2f} at our move {a['num']} ({a['mv']}), {b['q']:+.2f} at our move {b['num']} "
                    f"({b['mv']}).")
            with open(os.path.join(out_dir, stem + "_before.sgf"), "w", encoding="utf-8") as fh:
                fh.write(cut_sgf(text, a["num"] - 1, f"{desc} Before our move {a['num']} (we played {a['mv']})."))
            with open(os.path.join(out_dir, stem + "_after.sgf"), "w", encoding="utf-8") as fh:
                fh.write(cut_sgf(text, b["num"] - 1, f"{desc} After the opponent's reply, before our move "
                                                      f"{b['num']} (we played {b['mv']})."))
            index.append(f"{stem}: {desc}")
    open(os.path.join(out_dir, "index.txt"), "w", encoding="utf-8").write("\n".join(index) + "\n")
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default=os.path.join(ROOT, "runs/cgos/log/cgos.log"))
    ap.add_argument("--moves", default=os.path.join(ROOT, "runs/cgos/log/moves.log"))
    ap.add_argument("--sgf", default=os.path.join(ROOT, "runs/cgos/sgf"))
    ap.add_argument("--blindspots", default=None, help="folder for blind-spot positions as SGFs")
    ap.add_argument("--drop", type=float, default=0.25)
    args = ap.parse_args()
    games = [g for g in client_games(args.client) if "result" in g]
    moves = engine_moves(args.moves) if os.path.exists(args.moves) else []
    if args.blindspots:
        k = write_blindspots(args.blindspots, games, moves, args.sgf, args.drop)
        print(f"wrote {k} blind-spot positions (x2 SGFs) to {args.blindspots}\n")
    print(f"{len(games)} finished CGOS games\n")
    rows = []
    for g in games:
        mv = [m for m in moves if g["start"] <= m["t"] <= g["end"]]
        won = g.get("won")
        line = (f"{g['start']:%a %H:%M}  {'draw' if won == 0.5 else ('WIN ' if won else 'loss')}  {g['result']:>8s}  as {g['color']:5s} vs "
                f"{g['opp']} ({g['opp_rating']})  [our rating then {g['our_rating']}]")
        print(line)
        rows.append((g, won))
        if not mv:
            print("    (no engine log for this game)\n")
            continue
        used = sum(m["time"] for m in mv)
        reuse = sum(m["reused"] for m in mv) / max(1, sum(m["visits"] for m in mv))
        print(f"    {len(mv)} own moves, clock used {used:.0f} s ({used / 3:.0f}%), mean visits "
              f"{sum(m['visits'] for m in mv) / len(mv):.0f}, reused {100 * reuse:.0f}%, "
              f"extended {100 * sum(m['ext'] for m in mv) / len(mv):.0f}%")
        qs = [m["q"] for m in mv]
        marks = [mv[min(len(mv) - 1, k)] for k in (4, 9, 19, 29)]
        print("    Q at our moves " + ", ".join(f"#{m['num']}: {m['q']:+.2f}" for m in marks)
              + f"; min {min(qs):+.2f}, final {qs[-1]:+.2f}")
        drops = sorted(((a["q"] - b["q"], a, b) for a, b in zip(mv, mv[1:])), key=lambda x: -x[0])[:3]
        drops = [d for d in drops if d[0] >= 0.10]
        if drops:
            print("    biggest drops: " + "; ".join(f"{a['q']:+.2f} -> {b['q']:+.2f} between our move #{a['num']} "
                                                     f"({a['mv']}) and #{b['num']} ({b['mv']})" for _, a, b in drops))
        lost_from = next((m for m in mv if m["q"] < -0.5), None)
        if lost_from and not won:
            print(f"    first Q below -0.5 at our move #{lost_from['num']} ({lost_from['mv']})")
        passes = [m["num"] for m in mv if m["mv"] == "pass"]
        if len(passes) >= 3:
            print(f"    passed {len(passes)} times while the game went on (moves {passes[0]}-{passes[-1]})")
        print()
    if rows:
        print("Totals")
        for c in ("black", "white"):
            r = [w for g, w in rows if g["color"] == c]
            print(f"  as {c}: {sum(r)}/{len(r)}")
        for lo, hi in ((0, 2200), (2200, 2600), (2600, 2900), (2900, 9999)):
            r = [w for g, w in rows if lo <= g["opp_rating"] < hi]
            if r:
                print(f"  vs opponents rated {lo}-{hi if hi < 9999 else '':}: {sum(r)}/{len(r)}")


if __name__ == "__main__":
    sys.exit(main())
