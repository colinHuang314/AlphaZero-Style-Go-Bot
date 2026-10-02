"""Component-level diagnostics for a checkpoint (new GoNet or original AZNet).

    python tools/diagnose.py MODEL.pt [--val runs/9x9_a/val_buffer.npz] [--positions 200]

Reports:
  1. symmetry consistency: std of value / top-move agreement across the 8 board
     symmetries (a well-trained net should be nearly invariant)
  2. held-out accuracy (GoNet + val buffer only): policy top-1 vs MCTS target,
     value accuracy and calibration table, ownership error
  3. tactics suite: hand-built positions with a known correct move; reports the raw
     policy's top move and the MCTS choice at a few visit counts
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gozero.go.state import GoState, move_to_str, str_to_move  # noqa: E402
from gozero.loop import load_anchor  # noqa: E402
from gozero.mcts.evaluator import NetEvaluator  # noqa: E402
from gozero.mcts.mcts import BatchedMCTS, MCTSConfig, Node, pick_move  # noqa: E402

# ------------------------------------------------------------------ tactics
# (name, black stones, white stones, to_play, correct moves, note)
TACTICS_9 = [
    ("capture in 1", ["D4", "E5", "F4"], ["E4"], 1, ["E3"], "white E4 has one liberty"),
    ("escape atari", ["E4"], ["D4", "F4", "E5"], 1, ["E3"], "extend the atari'd stone"),
    ("capture 4 stones", ["C5", "D6", "E6", "F5", "C4", "F4", "D3"], ["D5", "E5", "D4", "E4"], 1, ["E3"],
     "4-stone white group, last liberty E3"),
    ("save group with 1 liberty", ["C3", "D3", "E3"], ["B3", "C4", "D4", "E4", "F3", "C2", "D2"], 1, ["E2"],
     "black C3-E3 in atari, E2 only extension"),
]


def build(n, black, white, to_play):
    s = GoState.new(n)
    b = s.board.copy()
    for m in black:
        b[str_to_move(m, n)] = 1
    for m in white:
        b[str_to_move(m, n)] = -1
    return GoState(s.rules, b, to_play)


def own_play_positions(ev, n, count, rng):
    """Positions from the model's own (policy-sampled) play: in-distribution for that model."""
    out = []
    while len(out) < count:
        s = GoState.new(n)
        target = int(rng.integers(0, 2 * n * n // 2))
        while s.move_number < target and not s.is_terminal():
            o = ev.raw([s], 0)
            p = np.exp(o["policy"][0] - o["policy"][0].max()) * s.legal_mask()
            s = s.play(int(rng.choice(len(p), p=p / p.sum())))
        if not s.is_terminal():
            out.append(s)
    return out


def symmetry_report(ev, positions):
    vstd, agree, pl1 = [], [], []
    for s in positions:
        outs = [ev.raw([s], k) for k in range(8)]
        if "value" in outs[0]:
            v = np.array([o["value"][0] for o in outs])
        else:
            v = np.tanh(np.array([o["value_logit"][0] for o in outs]) / 2)
        vstd.append(v.std())
        mask = np.log(s.legal_mask() + 1e-12)
        ps = []
        for o in outs:
            x = o["policy"][0] + mask
            e = np.exp(x - x.max())
            ps.append(e / e.sum())
        tops = [int(np.argmax(p)) for p in ps]
        agree.append(np.mean([t == tops[0] for t in tops[1:]]))
        pl1.append(np.mean([np.abs(p - ps[0]).sum() for p in ps[1:]]))
    print(f"[symmetry] value std across symmetries: {np.mean(vstd):.3f} (0 = perfectly consistent)")
    print(f"[symmetry] top-move agreement: {100 * np.mean(agree):.0f}%   policy L1 difference: {np.mean(pl1):.2f}")


def heldout_report(model, val_path, device):
    from gozero.train.replay import ReplayBuffer
    from gozero.train.trainer import TrainConfig, compute_losses, to_tensors
    buf = ReplayBuffer.load(val_path)
    idx = buf.recent_indices(min(buf.size, 8192))
    probs, zs, stats_acc = [], [], {}
    with torch.no_grad():
        for i in range(0, len(idx), 1024):
            t = to_tensors(buf.gather(idx[i:i + 1024]), device)
            _, st = compute_losses(model, t, TrainConfig())
            for k, v in st.items():
                stats_acc[k] = stats_acc.get(k, 0) + v * len(idx[i:i + 1024])
            probs.append(torch.sigmoid(model(t["features"])["value_logit"]).cpu().numpy())
            zs.append(t["z"].cpu().numpy())
    for k, v in stats_acc.items():
        print(f"[held-out] {k}: {v / len(idx):.3f}")
    p, z = np.concatenate(probs), np.concatenate(zs)
    print("[held-out] value calibration (predicted win prob -> actual win rate, count):")
    for lo in np.arange(0, 1, 0.1):
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum() >= 20:
            print(f"    {lo:.1f}-{lo + 0.1:.1f}: {np.mean(z[m] > 0):.2f}  (n={m.sum()})")


def tactics_report(ev, n, visits_list=(1, 32, 200)):
    if n != 9:
        print("[tactics] suite is 9x9 only")
        return
    mcts = BatchedMCTS(MCTSConfig(dirichlet_eps=0.0), np.random.default_rng(0))
    score = {v: 0 for v in ("policy",) + tuple(visits_list)}
    total = 0
    for name, black, white, tp, good, note in TACTICS_9:
        s = build(n, black, white, tp)
        gs = {str_to_move(m, n) for m in good}
        ok = lambda m, gs=gs: m in gs
        total += 1
        o = ev.raw([s], 0)
        pol = o["policy"][0] + np.log(s.legal_mask() + 1e-12)
        top = int(np.argmax(pol))
        line = f"  {name:32s} policy:{move_to_str(top, n):>4s}{'+' if ok(top) else 'x'}"
        score["policy"] += ok(top)
        for v in visits_list:
            root = Node(s)
            mcts.search([root], ev, v)
            m = pick_move(root, 0, np.random.default_rng(0))
            score[v] += ok(m)
            line += f"  {v}v:{move_to_str(m, n):>4s}{'+' if ok(m) else 'x'}"
        print(line)
    print("[tactics] " + ", ".join(f"{k}: {v}/{total}" for k, v in score.items()))


# ------------------------------------------------------------------ life and death
# Classic eye shapes for an enclosed white group (white has no outside liberties).
# Each entry: name, black, white, expected liberties of the white group (a self-check
# that the position is what it claims), then the probes to run on it.
_STRAIGHT3_B = ["A1", "A2", "A3", "B3", "C3", "D3", "E3", "F3", "G3", "G2", "G1"]
_STRAIGHT3_W = ["B1", "B2", "C2", "D2", "E2", "F2", "F1"]
_BENT3_B = ["D1", "D2", "D3", "C3", "C4", "B4", "A4"]
_BENT3_W = ["C1", "C2", "B2", "B3", "A3"]
_SQUARE4_B = ["D1", "D2", "D3", "D4", "C4", "B4", "A4"]
_SQUARE4_W = ["C1", "C2", "C3", "B3", "A3"]
_TWO_EYES_B = ["A3", "B3", "C3", "D3", "E3", "E2", "E1"]
_TWO_EYES_W = ["B1", "A2", "B2", "C2", "D2", "D1"]

LIFE_DEATH_9 = [
    # (name, black, white, white-group liberties, to_play, correct moves)
    ("straight three: black kills", _STRAIGHT3_B, _STRAIGHT3_W, ["C1", "D1", "E1"], 1, ["D1"]),
    ("straight three: white lives", _STRAIGHT3_B, _STRAIGHT3_W, ["C1", "D1", "E1"], -1, ["D1"]),
    ("bent three (corner): black kills", _BENT3_B, _BENT3_W, ["A2", "B1"], 1, ["A1"]),
    ("bent three (corner): white lives", _BENT3_B, _BENT3_W, ["A2", "B1"], -1, ["A1"]),
]
# Groups whose status doesn't depend on who moves: does the ownership head see it?
# expected: +1 = the white stones will end up black's (dead), -1 = stay white's (alive)
STATUS_9 = [
    ("square four (dead either way)", _SQUARE4_B, _SQUARE4_W, ["A2", "B2", "B1"], +1),
    ("two real eyes (alive)", _TWO_EYES_B, _TWO_EYES_W, ["A1", "C1"], -1),
]


def _check_libs(s, white, libs, n):
    gid, glibs = s.groups()
    g = gid[str_to_move(white[0], n)]
    members = {p for p in range(n * n) if gid[p] == g}
    assert members == {str_to_move(m, n) for m in white}, "white stones are not one group"
    nb = s.rules.nb
    got = {int(q) for p in members for q in nb[p] if q >= 0 and s.board[q] == 0}
    assert got == {str_to_move(m, n) for m in libs}, f"liberties {sorted(move_to_str(q, n) for q in got)}"


def life_death_report(ev, n, visits_list=(1, 64, 400)):
    if n != 9:
        return
    mcts = BatchedMCTS(MCTSConfig(dirichlet_eps=0.0), np.random.default_rng(0))
    score = {v: 0 for v in ("policy",) + tuple(visits_list)}
    for name, black, white, libs, tp, good in LIFE_DEATH_9:
        s = build(n, black, white, tp)
        _check_libs(s, white, libs, n)
        gs = {str_to_move(m, n) for m in good}
        o = ev.raw([s], 0)
        top = int(np.argmax(o["policy"][0] + np.log(s.legal_mask() + 1e-12)))
        line = f"  {name:34s} policy:{move_to_str(top, n):>4s}{'+' if top in gs else 'x'}"
        score["policy"] += top in gs
        for v in visits_list:
            root = Node(s)
            mcts.search([root], ev, v)
            m = pick_move(root, 0, np.random.default_rng(0))
            score[v] += m in gs
            line += f"  {v}v:{move_to_str(m, n):>4s}{'+' if m in gs else 'x'}"
        print(line)
    print(f"[life&death] vital point found: " + ", ".join(f"{k}: {v}/{len(LIFE_DEATH_9)}" for k, v in score.items()))
    for name, black, white, libs, expect in STATUS_9:
        for tp in (1, -1):
            s = build(n, black, white, tp)
            _check_libs(s, white, libs, n)
            o = ev.raw([s], 0)
            idx = [str_to_move(m, n) for m in white]
            if "ownership" in o:
                own_black = float(np.mean(o["ownership"][0][idx])) * tp  # to black's perspective
                verdict = "dead" if own_black > 0.3 else "alive" if own_black < -0.3 else "unsure"
                right = (own_black > 0.3) == (expect > 0) and verdict != "unsure"
                own_txt = f"ownership of white stones {own_black:+.2f} (black +1) -> {verdict} {'+' if right else 'x'}"
            else:
                own_txt = "(no ownership head)"
            v_black = (float(o["value"][0]) if "value" in o else float(np.tanh(o["value_logit"][0] / 2))) * tp
            print(f"  {name:30s} {'B' if tp == 1 else 'W'} to play: {own_txt}; P(black wins) {100 * (v_black + 1) / 2:.0f}%")


# ------------------------------------------------------------------ problem files
from gozero.eval.problems import is_correct, load_problem  # noqa: E402  (shared with the UI)


def problems_report(ev, folder, max_visits=12800, seeds=3):
    """For each problem: the raw policy's top move under each of the 8 board symmetries,
    then `seeds` searches grown in doublings (100, 200, ... max_visits), recording the
    top move at each checkpoint. Searches use a random symmetry per batch, as in play,
    seeded per puzzle so results are repeatable and don't depend on puzzle order
    (seeds=0: old behaviour, one search with every evaluation in the original orientation).
    'needs' = the smallest checkpoint from which the top move stays correct, per search."""
    import glob
    from gozero.mcts.mcts import search_single
    files = sorted(glob.glob(os.path.join(folder, "*.txt")) + glob.glob(os.path.join(folder, "*.sgf")))
    cps = [100]
    while cps[-1] * 2 <= max_visits:
        cps.append(cps[-1] * 2)
    cfg = MCTSConfig(dirichlet_eps=0.0)
    runs = max(seeds, 1)
    solved = {c: 0.0 for c in ["policy"] + cps}
    total = 0
    print(f"  {'puzzle':44s} {'ans':>5s} {'policy':>8s} " + " ".join(f"{c:>6d}" for c in cps) +
          f"   needs ({runs} run{'s' if runs > 1 else ''})   %visits on answer @max")
    for f in files:
        try:
            p = load_problem(f)
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")
            continue
        s, n = p["state"], p["state"].rules.n
        ok = lambda m, p=p: is_correct(p, m)
        total += 1
        legal = np.log(s.legal_mask() + 1e-12)
        ptops = [int(np.argmax(ev.raw([s], k)["policy"][0] + legal)) for k in range(8)]
        pok = sum(ok(m) for m in ptops)
        solved["policy"] += pok / 8
        hits, needs_all, shares = np.zeros(len(cps), int), [], []
        for r in range(runs):
            ev.random_symmetry = seeds > 0
            ev.rng = np.random.default_rng(r)
            root, marks = Node(s), []
            for c in cps:
                search_single(root, ev, cfg, c, leaf_batch=16)
                marks.append(ok(int(root.moves[int(np.argmax(root.Nc))])))
            hits += marks
            needs_all.append(next((c for i, c in enumerate(cps) if all(marks[i:])), None))
            shares.append(sum(root.Nc[i] for i, m in enumerate(root.moves) if ok(int(m))) / root.Nc.sum())
        for c, h in zip(cps, hits):
            solved[c] += h / runs
        mark = (lambda h: "+" if h else "x") if runs == 1 else (lambda h: f"{h}/{runs}")
        needs = "/".join(">" + str(max_visits) if x is None else str(x) for x in needs_all)
        ans = "tenuki" if p["answers"] is None else ",".join(move_to_str(m, n) for m in sorted(p["answers"]))
        print(f"  {p['name'][:44]:44s} {ans:>5s} {move_to_str(ptops[0], n):>4s} {pok}/8 " +
              " ".join(f"{mark(h):>6s}" for h in hits) +
              f"   {needs:>18s}   {100 * np.mean(shares):5.1f}%")
    if total:
        fmt = lambda v: f"{v:.0f}" if v == int(v) else f"{v:.1f}"
        print(f"[problems] solved (avg over {runs} run{'s' if runs > 1 else ''}; policy over 8 symmetries): "
              f"policy {fmt(solved['policy'])}/{total}, " + ", ".join(f"{c}v {fmt(solved[c])}/{total}" for c in cps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--val", default=None)
    ap.add_argument("--positions", type=int, default=100)
    ap.add_argument("--board", type=int, default=9)
    ap.add_argument("--problems", default=None, help="folder of problem files; runs only the problem report")
    ap.add_argument("--max-visits", type=int, default=12800, help="largest search size for --problems")
    ap.add_argument("--seeds", type=int, default=3,
                    help="searches per puzzle, each with seeded random symmetry as in play (0 = one search, original orientation only)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = load_anchor(args.model, dev)
    ev.random_symmetry = False
    print(f"== {args.model}")
    if args.problems:
        problems_report(ev, args.problems, args.max_visits, args.seeds)
        return
    rng = np.random.default_rng(0)
    symmetry_report(ev, own_play_positions(ev, args.board, args.positions, rng))
    if args.val and isinstance(ev, NetEvaluator):
        heldout_report(ev.model, args.val, dev)
    tactics_report(ev, args.board)
    life_death_report(ev, args.board)


if __name__ == "__main__":
    main()
