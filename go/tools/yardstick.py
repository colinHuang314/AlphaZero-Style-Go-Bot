"""Measure a go-zero net against KataGo at a fixed visit count (a fixed-strength yardstick; KataGo is only
an opponent here, never training data, so go-zero stays pure zero).

    python tools/yardstick.py MODEL --kata-visits 16 [--pairs 50] [--visits 800]
    python tools/yardstick.py runs/eval_night6/candidate_c439.pt --calibrate

Games: tools/gtp_match.py, komi 7.5 (the training komi, so no draws and no komi effects), paired random
openings, our bot at --visits per move. --calibrate walks KataGo's visits up or down by 4x from --kata-visits
with short matches (--ladder-pairs) until our bot scores 35-65%, then plays --pairs there. Each result line
is appended to runs/yardstick/results.txt with a CGOS estimate: c439 was ~2630 on CGOS, so
KataGo@V ~= 2630 - Elo(c439 vs KataGo@V), and any net ~= KataGo@V + Elo(net vs KataGo@V).
KataGo needs tools/katago/ (katago.exe + the network; see configs/katago_yardstick.cfg).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "runs", "yardstick")
C439_CGOS = 2630


def kata_cmd(visits):
    net = sorted(glob.glob(os.path.join(ROOT, "tools", "katago", "*.bin.gz")))[-1]
    # Windows CreateProcess doesn't find a relative exe path against Popen's cwd: give it the absolute one
    exe = os.path.join(ROOT, "tools", "katago", "katago.exe")
    return (f"{exe} gtp -config configs/katago_yardstick.cfg -model {os.path.relpath(net, ROOT).replace(os.sep, '/')}"
            f" -override-config maxVisits={visits}")


def match(model, kv, pairs, visits, seed):
    tag = f"{os.path.splitext(os.path.basename(model))[0]}_v{visits}_vs_kata{kv}_p{pairs}_s{seed}"
    ours = f"python -u -m gozero.gtp --model {model} --visits {visits} --quiet".replace("\\", "/")
    cmd = [sys.executable, "-u", "tools/gtp_match.py", "--a", ours, "--b", kata_cmd(kv), "--pairs", str(pairs),
           "--komi", "7.5", "--seed", str(seed), "--sgf", f"runs/yardstick/{tag}"]
    t = time.time()
    with open(os.path.join(OUT, tag + ".txt"), "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    text = open(os.path.join(OUT, tag + ".txt"), encoding="utf-8").read()
    m = re.search(r": ([\d.]+)-([\d.]+) \(([\d.]+)%.*?Elo ([+-]\d+) \[([+-]\d+), ([+-]\d+)\]", text)
    if rc or not m:
        raise SystemExit(f"match failed (exit {rc}); see runs/yardstick/{tag}.txt")
    a, b, pct, elo, lo, hi = m.groups()
    line = (f"{time.strftime('%a %b %d %H:%M')}  {os.path.basename(model)} @{visits} vs KataGo@{kv}: {a}-{b} "
            f"({pct}%, Elo {elo} [{lo}, {hi}]), {time.time() - t:.0f}s")
    with open(os.path.join(OUT, "results.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return float(pct), int(elo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--kata-visits", type=int, default=16)
    ap.add_argument("--visits", type=int, default=800, help="our bot's visits per move")
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ladder-pairs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    kv = args.kata_visits
    if args.calibrate:
        seen = set()
        while kv not in seen:
            seen.add(kv)
            pct, _ = match(args.model, kv, args.ladder_pairs, args.visits, args.seed)
            if 35 <= pct <= 65:
                break
            nxt = kv * 4 if pct > 65 else max(1, kv // 4)
            if nxt == kv:
                break
            kv = nxt
    pct, elo = match(args.model, kv, args.pairs, args.visits, args.seed + 1)
    if args.calibrate:
        kata = C439_CGOS - elo
        note = f"calibration: KataGo@{kv} ~ {kata} on CGOS (from {os.path.basename(args.model)} ~ {C439_CGOS})"
    else:
        note = f"estimate needs a calibrated KataGo@{kv} rating R: {os.path.basename(args.model)} ~ R {elo:+d}"
    with open(os.path.join(OUT, "results.txt"), "a", encoding="utf-8") as f:
        f.write("  " + note + "\n")
    print(note)


if __name__ == "__main__":
    main()
