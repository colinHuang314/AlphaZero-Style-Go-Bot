"""State and background analysis behind the web UI.

One `UIEngine` owns the current game (a linear move list with a cursor), the
loaded models, and a worker thread that keeps searching the current position
(analysis mode) or plays bot moves (play / watch modes). HTTP handlers call
`action()` and `snapshot()`; everything shared is guarded by one lock.
"""
import glob
import os
import threading
import time

import numpy as np
import torch

from ..go.state import GoState, default_komi
from ..legacy.old_net import load_legacy
from ..mcts.evaluator import LegacyEvaluator, NetEvaluator
from ..mcts.mcts import MCTSConfig, Node, pick_move, principal_variation, search_single
from ..nn.model import load_model
from ..selfplay.selfplay import GameRecord, to_sgf
from .puzzles import PuzzleBench

MODES = ("analyze", "play", "watch", "policy", "sandbox", "puzzle")


class LoadedModel:
    def __init__(self, path, device, name=None):
        self.path = name or path  # display / cache key (relative to the project root)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "net_config" in ck:
            model, _ = load_model(path, device)
            self.ev = NetEvaluator(model, seed=0)
            self.kind = "gonet"
            self.n = model.cfg.board_size
            self.params = sum(p.numel() for p in model.parameters())
        else:
            net = load_legacy(path, device)
            self.ev = LegacyEvaluator(net, seed=0)
            self.kind = "legacy"
            self.n = net.board_size
            self.params = sum(p.numel() for p in net.parameters())

    def raw(self, states):
        """Policy probs (N+1), value (side to move), and optional score / ownership."""
        out = self.ev.raw(states, 0)
        res = []
        for i, s in enumerate(states):
            logits = out["policy"][i] + np.log(s.legal_mask() + 1e-12)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            if self.kind == "gonet":
                v = float(np.tanh(out["value_logit"][i] / 2))
                res.append({"policy": p, "value": v, "score": float(out["score"][i]),
                            "ownership": out["ownership"][i]})
            else:
                res.append({"policy": p, "value": float(out["value"][i]), "score": None, "ownership": None})
        return res


def list_models(root):
    pats = ["runs/eval_*/*.pt", "runs/*/latest.pt", "runs/*/models/*.pt", "anchors/*.pt"]
    found = []
    for p in pats:
        for f in sorted(glob.glob(os.path.join(root, p)), reverse="models" in p):
            rel = os.path.relpath(f, root).replace("\\", "/")
            if rel not in found:
                found.append(rel)
    return found


class UIEngine:
    def __init__(self, root_dir, default_model=None, device=None):
        self.root_dir = root_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.RLock()
        self.models = {}
        self.mcts_cfg = MCTSConfig(dirichlet_eps=0.0)
        self.settings = {
            "mode": "analyze",
            "model": None,          # analysis / play model
            "white_model": None,    # watch mode: white's model (black uses "model")
            "human_color": 1,       # play mode
            "bot_visits": 800,
            "watch_visits": 400,
            "max_visits": 20000,    # analysis stops here
            "analyzing": True,
            "bot_delay": 0.4,
        }
        self.new_game(9)
        self.puzzle = PuzzleBench(self, os.path.join(root_dir, "problems"))
        models = list_models(root_dir)
        pick = default_model or next((m for m in models if "candidate" in m), None) or (models[0] if models else None)
        if pick:
            self.set_model(pick)
            self.settings["white_model"] = next((m for m in models if "AZNET9_epoch_300" in m), pick)
        self.vps = 0.0
        self.eval_count = 0
        self.message = ""
        self._stop = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------ game state
    def new_game(self, n):
        with self.lock:
            self.n = n
            self.states = [GoState.new(n)]
            self.moves = []
            self.cursor = 0
            self.trees = {}       # (cursor, model path) -> Node
            self.netinfo = {}     # (cursor, model path) -> raw net output
            self.graph = {}       # cursor -> {"net": black winrate, "mcts": black winrate, "visits": n}
            self.message = ""

    @property
    def state(self):
        return self.states[self.cursor]

    def get_model(self, path):
        if path is None:
            return None
        if path not in self.models:
            self.models[path] = LoadedModel(os.path.join(self.root_dir, path), self.device, name=path)
        return self.models[path]

    def set_model(self, path):
        m = self.get_model(path)
        with self.lock:
            self.settings["model"] = path
            if m.n != self.n:
                self.new_game(m.n)

    def model_for_turn(self):
        """Which model searches the current position."""
        s = self.settings
        if s["mode"] == "watch" and self.state.to_play == -1:
            return self.get_model(s["white_model"])
        return self.get_model(s["model"])

    def _truncate(self):
        """Drop every move after the cursor (and all cached analysis of those positions)."""
        self.moves = self.moves[:self.cursor]
        self.states = self.states[:self.cursor + 1]
        for d in (self.trees, self.netinfo):
            for k in [k for k in d if k[0] > self.cursor]:
                del d[k]
        for k in [k for k in self.graph if k > self.cursor]:
            del self.graph[k]

    def play(self, move):
        with self.lock:
            st = self.state
            if st.is_terminal():
                return False, "game is over"
            if not st.legal_mask()[move]:
                return False, "illegal move"
            if self.cursor < len(self.moves) and self.moves[self.cursor] != move:
                self._truncate()  # branching off: the old continuation is discarded
            if self.cursor == len(self.moves):
                self.moves.append(move)
                self.states.append(st.play(move))
                # carry every model's search tree into the new position
                for (c, path), tree in list(self.trees.items()):
                    if c == self.cursor and tree.expanded:
                        self.trees[(c + 1, path)] = tree.child_by_move(move)
            self.cursor += 1
            self.message = ""
            return True, ""

    def goto(self, idx):
        with self.lock:
            self.cursor = int(max(0, min(len(self.moves), idx)))

    # ------------------------------------------------------------ analysis
    def _tree(self, model):
        key = (self.cursor, model.path)
        t = self.trees.get(key)
        if t is None:
            t = Node(self.state)
            self.trees[key] = t
        return t

    def _net(self, model, cursor):
        key = (cursor, model.path)
        if key not in self.netinfo:
            self.netinfo[key] = model.raw([self.states[cursor]])[0]
        return self.netinfo[key]

    def _fill_graph(self, model):
        """Raw-net win rate for every position (one batch), so the graph is never empty."""
        missing = [i for i in range(len(self.states)) if (i, model.path) not in self.netinfo]
        if not missing:
            return
        outs = model.raw([self.states[i] for i in missing])
        for i, o in zip(missing, outs):
            self.netinfo[(i, model.path)] = o

    def _run(self):
        last_t, last_visits = time.time(), 0
        while not self._stop:
            try:
                did = self._tick()
            except Exception as e:  # keep the UI alive, report the error
                self.message = f"engine error: {e!r}"
                did = False
            now = time.time()
            if now - last_t > 0.5:
                self.vps = (self.eval_count - last_visits) / (now - last_t)
                last_t, last_visits = now, self.eval_count
            # always yield briefly: Python locks aren't fair, and without this pause the
            # search thread re-acquires the lock immediately and starves the HTTP handlers
            time.sleep(0.03 if not did else 0.004)

    def _tick(self):
        s = self.settings
        mode = s["mode"]
        if mode == "puzzle":
            with self.lock:
                return self.puzzle.tick()
        with self.lock:
            model = self.model_for_turn()
            if model is None or mode == "sandbox":
                return False
            self._fill_graph(self.get_model(s["model"]))
            st = self.state
            if st.is_terminal():
                return False
            bots_turn = (mode == "watch") or (mode == "play" and st.to_play != s["human_color"])
            at_end = self.cursor == len(self.moves)
            if bots_turn and at_end:
                target = s["watch_visits"] if mode == "watch" else s["bot_visits"]
            elif mode in ("analyze", "play") and s["analyzing"]:
                target = s["max_visits"]
            else:
                return False
            tree = self._tree(model)
            if tree.N < target:
                self.eval_count += search_single(tree, model.ev, self.mcts_cfg, min(target, tree.N + 48), leaf_batch=16)
                if model.path == s["model"]:  # graph follows the primary model only
                    self._record_graph(tree)
                return True
            if bots_turn and at_end:
                move = pick_move(tree, 0.0, np.random.default_rng())
                self.play(move)
                time.sleep(s["bot_delay"] if mode == "watch" else 0.05)
                return True
            return False

    def _record_graph(self, tree):
        if tree.N < 1:
            return
        q = tree.q()  # side to move
        black = q if tree.state.to_play == 1 else -q
        self.graph[self.cursor] = {"mcts": (black + 1) / 2, "visits": int(tree.N)}

    # ------------------------------------------------------------ snapshot
    def snapshot(self):
        if self.settings["mode"] == "puzzle":
            with self.lock:
                out = self.puzzle.snapshot()
                out.update({"settings": dict(self.settings), "models": list_models(self.root_dir),
                            "message": self.message, "vps": self.vps})
                return out
        with self.lock:
            s = self.settings
            st = self.state
            n = self.n
            model = self.model_for_turn() if s["mode"] != "sandbox" else None
            out = {
                "n": n, "komi": st.rules.komi, "board": st.board.tolist(), "to_play": st.to_play,
                "cursor": self.cursor, "length": len(self.moves), "moves": self.moves,
                "last_move": self.moves[self.cursor - 1] if self.cursor > 0 else None,
                "passes": st.passes, "game_over": st.is_terminal(), "settings": dict(s),
                "models": list_models(self.root_dir), "message": self.message, "vps": self.vps,
                "legal": st.legal_mask().tolist(), "captures": self._captures(),
            }
            if st.is_terminal() or s["mode"] == "sandbox":
                own = st.ownership()
                out["territory"] = own.tolist()
                sc = st.score()
                out["score_now"] = sc
                if st.is_terminal():
                    out["result"] = f"{'B' if sc > 0 else 'W'}+{abs(sc)}"
            if model is not None:
                net = self._net(model, self.cursor)
                sign = st.to_play
                out["net"] = {
                    "policy": net["policy"].tolist(),
                    "black_winrate": (net["value"] * sign + 1) / 2,
                    "score_lead": None if net["score"] is None else net["score"] * sign,
                    "ownership": None if net["ownership"] is None else (net["ownership"] * sign).tolist(),
                    "model": model.path, "kind": model.kind,
                }
                tree = self.trees.get((self.cursor, model.path))
                if tree is not None and tree.expanded and tree.N > 0 and s["mode"] != "policy":
                    out["search"] = self._search_info(tree)
            base = self.get_model(s["model"]) if s["model"] else None
            if base is not None:
                g = []
                for i in range(len(self.states)):
                    info = self.netinfo.get((i, base.path))
                    net_b = None
                    if info is not None:
                        net_b = (info["value"] * self.states[i].to_play + 1) / 2
                    gm = self.graph.get(i, {})
                    g.append({"net": net_b, "mcts": gm.get("mcts"), "visits": gm.get("visits", 0)})
                out["graph"] = g
            return out

    def _search_info(self, tree):
        st = tree.state
        order = np.argsort(-tree.Nc)
        cands = []
        total = tree.Nc.sum()
        for i in order[:20]:
            if tree.Nc[i] < 1:
                break
            q = tree.Wc[i] / tree.Nc[i]  # for side to move
            cands.append({"move": int(tree.moves[i]), "visits": int(tree.Nc[i]), "share": float(tree.Nc[i] / total),
                          "winrate": (q + 1) / 2, "prior": float(tree.P_raw[i]),
                          "pv": principal_variation(tree, int(i))})
        q = tree.q()
        return {"visits": int(tree.N), "winrate": (q + 1) / 2,
                "black_winrate": ((q if st.to_play == 1 else -q) + 1) / 2, "candidates": cands}

    def _captures(self):
        """Stones captured by each color so far (along the current line)."""
        st = self.state
        placed = {1: 0, -1: 0}
        color = 1
        for m in self.moves[:self.cursor]:
            if m != self.n * self.n:
                placed[color] += 1
            color = -color
        on_board = {1: int((st.board == 1).sum()), -1: int((st.board == -1).sum())}
        # captures BY black = white stones that were placed but are gone
        return {"black": placed[-1] - on_board[-1], "white": placed[1] - on_board[1]}

    # ------------------------------------------------------------ actions
    def action(self, a):
        t = a.get("type")
        ok, msg = True, ""
        if t.startswith("pz_"):
            with self.lock:
                try:
                    msg = self.puzzle.action(a)
                except Exception as e:
                    msg = f"{e}"
            self.message = msg
            return {"ok": not msg, "message": msg}
        if t == "play":
            ok, msg = self.play(int(a["move"]))
        elif t == "pass":
            ok, msg = self.play(self.n * self.n)
        elif t == "undo":
            self.goto(self.cursor - (2 if self.settings["mode"] == "play" and self.cursor >= 2 else 1))
        elif t == "redo":
            self.goto(self.cursor + 1)
        elif t == "goto":
            self.goto(int(a["index"]))
        elif t == "new":
            n = int(a.get("size", self.n))
            m = self.get_model(self.settings["model"]) if self.settings["model"] else None
            if m is not None and m.n != n and self.settings["mode"] != "sandbox":
                return {"ok": False, "message": f"model is {m.n}x{m.n}; pick a {n}x{n} model or use sandbox"}
            self.new_game(n)
        elif t == "set":
            key, val = a["key"], a["value"]
            if key == "model":
                self.set_model(val)
            elif key in self.settings:
                with self.lock:
                    if key == "white_model":
                        self.get_model(val)
                    self.settings[key] = type(self.settings[key])(val) if self.settings[key] is not None else val
                    if key == "mode":
                        # no hints on your own turn when playing the bot; analysis on elsewhere
                        self.settings["analyzing"] = val != "play"
                        if val == "play":
                            self._truncate()
            else:
                ok, msg = False, f"unknown setting {key}"
        elif t == "load_sgf":
            ok, msg = self.load_sgf(a["text"])
        else:
            ok, msg = False, f"unknown action {t}"
        if not ok:
            self.message = msg
        return {"ok": ok, "message": msg}

    def sgf(self):
        with self.lock:
            rec = GameRecord(moves=list(self.moves), samples=[])
            last = self.states[-1]
            rec.score = last.score()
            rec.winner = 1 if rec.score > 0 else -1
            return to_sgf(rec, self.n, default_komi(self.n), "go-zero UI")

    def load_sgf(self, text):
        import re
        m = re.search(r"SZ\[(\d+)\]", text)
        n = int(m.group(1)) if m else 9
        moves = re.findall(r";\s*([BW])\[([a-s]{0,2})\]", text)
        self.new_game(n)
        cols = "abcdefghijklmnopqrs"
        for _, mv in moves:
            move = n * n if mv in ("", "tt") else cols.index(mv[1]) * n + cols.index(mv[0])
            ok, msg = self.play(move)
            if not ok:
                return False, f"SGF move {len(self.moves) + 1} rejected: {msg}"
        return True, ""
