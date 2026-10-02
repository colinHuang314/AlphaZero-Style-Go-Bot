"""Puzzle workbench behind the UI's Puzzles tab.

Set up a position stone by stone (no captures, no turn order), mark the
correct answers, save it as a problem file (problems/*.txt, the format
tools/diagnose.py reads), and run any model on one puzzle or on all of them.
"""
import os

import numpy as np

from ..eval.problems import (is_correct, list_problems, load_problem, save_problem, slugify,
                             zero_liberty_groups)
from ..go.state import GoState, Rules, move_to_str
from ..mcts.mcts import Node, search_single


class PuzzleBench:
    def __init__(self, engine, folder, n=9):
        self.eng = engine
        self.folder = folder
        self.n = n
        self.clear()
        self.batch = None

    # ------------------------------------------------------------ editing
    def clear(self):
        self.board = np.zeros(self.n * self.n, dtype=np.int8)
        self.to_play = 1
        self.marks = set()
        self.tenuki = False
        self.name = ""
        self.note = ""
        self.file = None
        self._invalidate()

    def _invalidate(self):
        self.tree = None
        self.tree_model = None
        self.target = 0
        self._net_key = None
        self._net = None

    def state(self):
        return GoState(Rules.get(self.n), self.board.copy(), self.to_play)

    def set_stone(self, p, color):
        if self.board[p] == color:
            return
        self.board[p] = color
        if color != 0 and not self.tenuki:
            self.marks.discard(p)  # an answer must stay an empty point
        self._invalidate()

    def toggle_mark(self, p):
        if not self.tenuki and self.board[p] != 0:
            return "answers must be empty points"
        self.marks.symmetric_difference_update({p})
        return ""

    # ------------------------------------------------------------ files
    def save(self, name, note, overwrite):
        name = (name or "").strip()
        if not name:
            return "give the puzzle a name first"
        if not self.marks:
            return "mark at least one answer (or the inside points for a tenuki puzzle)"
        bad = zero_liberty_groups(self.board, self.n)
        if bad:
            return "stones with no liberties: " + " ".join(move_to_str(p, self.n) for p in bad)
        st = self.state()
        if not self.tenuki:
            illegal = [p for p in self.marks if not st.legal_mask()[p]]
            if illegal:
                return "answer is not a legal move: " + " ".join(move_to_str(p, self.n) for p in illegal)
        fname = self.file if (overwrite and self.file) else slugify(name) + ".txt"
        existing = set(list_problems(self.folder))
        if not (overwrite and self.file):
            base, k = fname[:-4], 2
            while fname in existing:
                fname = f"{base}_{k}.txt"
                k += 1
        save_problem(self.folder, fname, name, self.board, self.to_play, self.n, self.marks, self.tenuki, note)
        self.file, self.name, self.note = fname, name, note
        return ""

    def load(self, fname):
        p = load_problem(os.path.join(self.folder, fname), self.n)
        st = p["state"]
        if st.rules.n != self.n:
            return f"puzzle is {st.rules.n}x{st.rules.n}"
        self.board = st.board.copy()
        self.to_play = st.to_play
        self.tenuki = p["answers"] is None
        self.marks = set(p["inside"] if self.tenuki else p["answers"])
        self.name, self.note, self.file = p["name"], p["note"], fname
        self._invalidate()
        return ""

    def delete(self, fname):
        path = os.path.join(self.folder, fname)
        if os.path.exists(path) and fname in list_problems(self.folder):
            os.remove(path)
            if self.file == fname:
                self.file = None
        return ""

    # ------------------------------------------------------------ running
    def run(self, visits):
        bad = zero_liberty_groups(self.board, self.n)
        if bad:
            return "fix stones with no liberties first: " + " ".join(move_to_str(p, self.n) for p in bad)
        model = self.eng.get_model(self.eng.settings["model"])
        if model.n != self.n:
            return f"model is {model.n}x{model.n}"
        if self.tree is None or self.tree_model != model.path:
            self.tree = Node(self.state())
            self.tree_model = model.path
        self.target = int(visits)
        return ""

    def run_all(self, visits):
        files = list_problems(self.folder)
        if not files:
            return "no saved puzzles yet"
        self.batch = {"model": self.eng.settings["model"], "visits": int(visits), "queue": files,
                      "results": [], "current": None}
        return ""

    def stop(self):
        self.target = self.tree.N if self.tree is not None else 0
        if self.batch is not None:
            self.batch["queue"] = []
            self.batch["current"] = None

    def tick(self):
        """One chunk of search work. Returns True if work was done."""
        eng = self.eng
        if self.batch is not None and (self.batch["current"] is not None or self.batch["queue"]):
            b = self.batch
            model = eng.get_model(b["model"])
            if b["current"] is None:
                f = b["queue"].pop(0)
                try:
                    prob = load_problem(os.path.join(self.folder, f), self.n)
                except Exception as e:
                    b["results"].append({"file": f, "name": f, "error": str(e)})
                    return True
                b["current"] = (prob, Node(prob["state"]))
            prob, root = b["current"]
            eng.eval_count += search_single(root, model.ev, eng.mcts_cfg, min(b["visits"], root.N + 64), leaf_batch=16)
            if root.N >= b["visits"] or root.terminal:
                b["results"].append(self._score(prob, root, model))
                b["current"] = None
            return True
        if self.tree is not None and self.tree.N < self.target and not self.tree.terminal:
            model = eng.get_model(self.tree_model)
            eng.eval_count += search_single(self.tree, model.ev, eng.mcts_cfg,
                                            min(self.target, self.tree.N + 48), leaf_batch=16)
            return True
        return False

    def _score(self, prob, root, model):
        n = prob["state"].rules.n
        if not root.expanded or root.N == 0:
            return {"file": prob["file"], "name": prob["name"], "error": "no search"}
        share = root.Nc / root.Nc.sum()
        top = int(root.moves[int(np.argmax(root.Nc))])
        ok_mask = np.array([is_correct(prob, int(m)) for m in root.moves])
        raw = model.raw([prob["state"]])[0]["policy"]
        prior_ok = float(sum(raw[m] for m in range(len(raw)) if is_correct(prob, m) and prob["state"].legal_mask()[m]))
        return {"file": prob["file"], "name": prob["name"], "top": move_to_str(top, n),
                "correct": bool(is_correct(prob, top)), "share_correct": float(share[ok_mask].sum()),
                "prior_correct": prior_ok,
                "answers": "tenuki" if prob["answers"] is None else " ".join(move_to_str(m, n) for m in sorted(prob["answers"]))}

    # ------------------------------------------------------------ snapshot
    def net_info(self, model):
        key = (self.board.tobytes(), self.to_play, model.path)
        if key != self._net_key:
            self._net = model.raw([self.state()])[0]
            self._net_key = key
        return self._net

    def snapshot(self):
        n = self.n
        st = self.state()
        out = {
            "n": n, "komi": st.rules.komi, "board": self.board.tolist(), "to_play": self.to_play,
            "cursor": 0, "length": 0, "moves": [], "last_move": None, "passes": 0, "game_over": False,
            "legal": st.legal_mask().tolist(), "captures": {"black": 0, "white": 0},
            "puzzle": {
                "name": self.name, "note": self.note, "file": self.file, "tenuki": self.tenuki,
                "marks": sorted(self.marks), "files": list_problems(self.folder),
                "bad": zero_liberty_groups(self.board, n), "target": self.target,
                "running": bool(self.tree is not None and self.tree.N < self.target),
                "batch": None,
            },
        }
        if self.batch is not None:
            b = self.batch
            done = [r for r in b["results"] if "correct" in r]
            out["puzzle"]["batch"] = {
                "model": b["model"], "visits": b["visits"], "results": b["results"],
                "remaining": len(b["queue"]) + (b["current"] is not None),
                "current": b["current"][0]["name"] if b["current"] is not None else None,
                "solved": sum(r["correct"] for r in done), "total": len(done),
            }
        model = self.eng.get_model(self.eng.settings["model"])
        if model is not None and model.n == n and not out["puzzle"]["bad"]:
            net = self.net_info(model)
            sign = self.to_play
            out["net"] = {
                "policy": net["policy"].tolist(), "black_winrate": (net["value"] * sign + 1) / 2,
                "score_lead": None if net["score"] is None else net["score"] * sign,
                "ownership": None if net["ownership"] is None else (net["ownership"] * sign).tolist(),
                "model": model.path, "kind": model.kind,
            }
            if self.tree is not None and self.tree.expanded and self.tree.N > 0 and self.tree_model == model.path:
                info = self.eng._search_info(self.tree)
                top = info["candidates"][0]["move"] if info["candidates"] else None
                if self.marks:
                    fake = {"answers": None if self.tenuki else set(self.marks),
                            "inside": set(self.marks) if self.tenuki else None}
                    info["top_correct"] = None if top is None else bool(is_correct(fake, top))
                    info["share_correct"] = float(sum(c["share"] for c in info["candidates"] if is_correct(fake, c["move"])))
                out["search"] = info
        return out

    # ------------------------------------------------------------ actions
    def action(self, a):
        t = a["type"]
        if t == "pz_stone":
            self.set_stone(int(a["point"]), int(a["color"]))
        elif t == "pz_mark":
            return self.toggle_mark(int(a["point"]))
        elif t == "pz_to_play":
            self.to_play = int(a["color"])
            self._invalidate()
        elif t == "pz_tenuki":
            self.tenuki = bool(a["value"])
            self.marks = set()
        elif t == "pz_meta":
            self.name, self.note = a.get("name", self.name), a.get("note", self.note)
        elif t == "pz_clear":
            self.clear()
        elif t == "pz_save":
            return self.save(a.get("name", ""), a.get("note", ""), bool(a.get("overwrite")))
        elif t == "pz_load":
            return self.load(a["file"])
        elif t == "pz_delete":
            return self.delete(a["file"])
        elif t == "pz_run":
            return self.run(a.get("visits", 800))
        elif t == "pz_run_all":
            return self.run_all(a.get("visits", 800))
        elif t == "pz_stop":
            self.stop()
        else:
            return f"unknown puzzle action {t}"
        return ""
