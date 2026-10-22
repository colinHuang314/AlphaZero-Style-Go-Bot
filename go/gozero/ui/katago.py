"""KataGo next to go-zero in the UI's Compare tab.

KataGo runs as its analysis engine (`katago analysis`, JSON lines on stdin/stdout) in a background process,
started the first time the Compare tab needs it. It is only displayed: nothing it says reaches training data,
replay buffers or saved puzzles, so go-zero stays pure zero.

A reader thread consumes KataGo's output the whole time; writing every query and only then reading would
deadlock once KataGo's stdout pipe filled. When the UI process dies, KataGo sees end-of-input and quits
(`-quit-without-waiting`).
"""
import glob
import json
import os
import subprocess
import threading

from ..go.state import move_to_str, str_to_move

# Our rules (area scoring, positional superko, suicide illegal), spelled out: KataGo's "tromp-taylor"
# shorthand allows multi-stone suicide.
RULES = {"ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE", "suicide": False, "hasButton": False,
         "whiteHandicapBonus": "0", "friendlyPassOk": False}


class KataGo:
    def __init__(self, root, config="configs/katago_ui.cfg", cmd=None):
        self.root = root
        if cmd is None:
            exe = os.path.join(root, "tools", "katago", "katago.exe")
            nets = sorted(glob.glob(os.path.join(root, "tools", "katago", "*.bin.gz")))
            if os.path.exists(exe) and nets:
                cmd = [exe, "analysis", "-config", os.path.join(root, config), "-model", nets[-1],
                       "-quit-without-waiting"]
        self.cmd = cmd
        self.available = cmd is not None
        self.net_name = os.path.basename(cmd[cmd.index("-model") + 1]) if cmd and "-model" in cmd else "KataGo"
        self.proc = None
        self.error = None
        self.results = {}       # query id -> latest response (final one once "done")
        self.lock = threading.Lock()
        self._next = 0

    # ------------------------------------------------------------ process
    def start(self):
        if not self.available:
            self.error = "KataGo not found in tools/katago/"
            return False
        if self.proc is not None and self.proc.poll() is None:
            return True
        try:
            self.proc = subprocess.Popen(self.cmd, cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                         creationflags=0x08000000 if os.name == "nt" else 0)   # no console window
        except OSError as e:
            self.error = f"could not start KataGo: {e}"
            return False
        self.error = None
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        return True

    def _read(self, proc):
        for line in proc.stdout:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" in r and "id" not in r:
                self.error = r["error"]
                continue
            qid = r.get("id")
            if qid is None or r.get("action") == "terminate":
                continue
            if "error" in r:
                r = {"error": r["error"], "done": True}
            else:
                r["done"] = not r.get("isDuringSearch", False)
            with self.lock:
                self.results[qid] = r
        if self.error is None:
            self.error = f"KataGo exited (code {proc.poll()})"

    def _send(self, obj):
        if not self.start():
            return False
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
            return True
        except OSError as e:
            self.error = f"KataGo pipe: {e}"
            return False

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None

    # ------------------------------------------------------------ queries
    def query(self, moves, n, komi, visits, report_every=0.3):
        """Analyse the position after `moves` (our move indices, Black first). Returns a query id."""
        with self.lock:
            self._next += 1
            qid = f"q{self._next}"
        colors = "BW"
        q = {"id": qid, "moves": [[colors[i % 2], move_to_str(m, n)] for i, m in enumerate(moves)],
             "rules": RULES, "komi": float(komi), "boardXSize": n, "boardYSize": n, "maxVisits": int(visits),
             "includePolicy": True}
        if report_every:   # stream partial results (live view); 0 = only the final one
            q["reportDuringSearchEvery"] = report_every
        if not self._send(q):
            with self.lock:
                self.results[qid] = {"error": self.error, "done": True}
        return qid

    def terminate(self, qid):
        res = self.result(qid)
        if res is None or not res.get("done"):
            self._send({"id": f"stop-{qid}", "action": "terminate", "terminateId": qid})

    def result(self, qid):
        with self.lock:
            return self.results.get(qid)

    def forget(self, qid):
        with self.lock:
            self.results.pop(qid, None)


def parse(res, n, to_play):
    """A KataGo response -> the UI's shape: win rates for the side to move (as KataGo reports them with
    reportAnalysisWinratesAs = SIDETOMOVE) plus Black's view; moves as our indices."""
    if res is None:
        return None
    if "error" in res:
        return {"error": res["error"], "done": True}
    ri = res.get("rootInfo", {})
    wr = ri.get("winrate", 0.5)
    cands = []
    for m in sorted(res.get("moveInfos", []), key=lambda m: m["order"]):
        cands.append({"move": str_to_move(m["move"], n), "visits": int(m["visits"]), "winrate": m["winrate"],
                      "prior": m["prior"], "score_lead": m.get("scoreLead"),
                      "pv": [str_to_move(x, n) for x in m.get("pv", [])]})
    total = sum(c["visits"] for c in cands) or 1
    for c in cands:
        c["share"] = c["visits"] / total
    policy = res.get("policy")
    lead = ri.get("scoreLead", 0.0)
    return {"visits": int(ri.get("visits", 0)), "winrate": wr, "black_winrate": wr if to_play == 1 else 1 - wr,
            "score_lead": lead, "black_score_lead": lead if to_play == 1 else -lead, "candidates": cands,
            "policy": [max(p, 0.0) for p in policy] if policy else None, "done": res.get("done", False)}


def compare(ours, kata, net_policy):
    """Where go-zero and KataGo agree on one position. `ours`: the UI's search info (side-to-move win rates),
    `kata`: parse() output, `net_policy`: our raw policy over N*N+1 moves."""
    if not ours or not ours.get("candidates") or not kata or not kata.get("candidates"):
        return None
    our_top, kata_top = ours["candidates"][0], kata["candidates"][0]
    by_move = {c["move"]: c for c in kata["candidates"]}
    out = {"agree": our_top["move"] == kata_top["move"], "our_top": our_top["move"], "kata_top": kata_top["move"]}
    k = by_move.get(our_top["move"])
    if k is not None:   # KataGo's view of our choice
        out["kata_on_ours_winrate"] = k["winrate"]
        out["winrate_loss"] = kata_top["winrate"] - k["winrate"]
        if k.get("score_lead") is not None and kata_top.get("score_lead") is not None:
            out["points_loss"] = kata_top["score_lead"] - k["score_lead"]
        out["kata_visits_on_ours"] = k["visits"]
    if net_policy is not None:
        p = net_policy[kata_top["move"]]
        out["our_prior_on_kata_top"] = float(p)
        out["our_rank_of_kata_top"] = int(sum(1 for x in net_policy if x > p)) + 1
    ours_by_move = {c["move"]: c for c in ours["candidates"]}
    o = ours_by_move.get(kata_top["move"])
    out["our_share_on_kata_top"] = o["share"] if o else 0.0
    out["our_winrate_on_kata_top"] = o["winrate"] if o else None
    return out
