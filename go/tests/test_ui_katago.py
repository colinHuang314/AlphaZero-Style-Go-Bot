"""The UI's KataGo wrapper (gozero/ui/katago.py), against a fake analysis engine: no KataGo or GPU needed."""
import sys
import textwrap
import time

from gozero.go.state import str_to_move
from gozero.ui.katago import KataGo, compare, parse

FAKE = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("action") == "terminate":
            print(json.dumps({"id": q["id"], "action": "terminate", "terminateId": q["terminateId"]}), flush=True)
            continue
        n = q["boardXSize"]
        assert q["rules"]["suicide"] is False and q["komi"] == 7.5
        to_move = "B" if len(q["moves"]) % 2 == 0 else "W"
        info = [{"move": "D4", "visits": 30, "winrate": 0.7, "scoreLead": 2.5, "prior": 0.4, "order": 0,
                 "pv": ["D4", "E5"]},
                {"move": "E5", "visits": 10, "winrate": 0.6, "scoreLead": 1.0, "prior": 0.3, "order": 1, "pv": ["E5"]}]
        root = {"winrate": 0.7, "scoreLead": 2.5, "visits": q["maxVisits"], "currentPlayer": to_move,
                "seenMoves": [m[1] for m in q["moves"]]}
        if q.get("reportDuringSearchEvery"):
            print(json.dumps({"id": q["id"], "isDuringSearch": True, "rootInfo": dict(root, visits=1),
                              "moveInfos": info, "turnNumber": len(q["moves"])}), flush=True)
        pol = [0.01] * (n * n + 1)
        pol[0] = -1.0
        print(json.dumps({"id": q["id"], "isDuringSearch": False, "rootInfo": root, "moveInfos": info,
                          "policy": pol, "turnNumber": len(q["moves"])}), flush=True)
''')


def _wait(k, qid, done=True, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        r = k.result(qid)
        if r is not None and (r.get("done") or not done):
            return r
        time.sleep(0.02)
    raise AssertionError("no answer from the fake KataGo")


def test_query_and_parse(tmp_path):
    fake = tmp_path / "fake_kata.py"
    fake.write_text(FAKE)
    k = KataGo(str(tmp_path), cmd=[sys.executable, "-u", str(fake)])
    try:
        n = 9
        moves = [str_to_move("E5", n), str_to_move("C3", n), n * n]   # B E5, W C3, B pass: White to move
        qid = k.query(moves, n, 7.5, 200)
        r = _wait(k, qid)
        assert r["rootInfo"]["seenMoves"] == ["E5", "C3", "pass"]
        info = parse(r, n, to_play=-1)
        assert info["done"] and info["visits"] == 200
        assert abs(info["winrate"] - 0.7) < 1e-9 and abs(info["black_winrate"] - 0.3) < 1e-9
        assert info["black_score_lead"] == -2.5
        assert [c["move"] for c in info["candidates"]] == [str_to_move("D4", n), str_to_move("E5", n)]
        assert abs(info["candidates"][0]["share"] - 0.75) < 1e-9
        assert info["policy"][0] == 0.0 and len(info["policy"]) == n * n + 1   # illegal (-1) -> 0
        k.terminate(qid)   # already done: nothing sent, nothing breaks
        q2 = k.query([], n, 7.5, 50, report_every=0)
        assert _wait(k, q2)["rootInfo"]["seenMoves"] == []
    finally:
        k.close()


def test_missing_katago(tmp_path):
    k = KataGo(str(tmp_path))
    assert not k.available
    qid = k.query([], 9, 7.5, 10)
    assert "error" in k.result(qid)


def test_compare():
    n = 9
    d4, e5, c3 = (str_to_move(m, n) for m in ("D4", "E5", "C3"))
    kata = {"candidates": [{"move": d4, "visits": 30, "winrate": 0.7, "score_lead": 2.5, "share": 0.75},
                           {"move": e5, "visits": 10, "winrate": 0.6, "score_lead": 1.0, "share": 0.25}]}
    ours = {"candidates": [{"move": e5, "visits": 80, "winrate": 0.55, "share": 0.8},
                           {"move": d4, "visits": 20, "winrate": 0.5, "share": 0.2}]}
    pol = [0.0] * (n * n + 1)
    pol[e5], pol[d4], pol[c3] = 0.5, 0.1, 0.2
    a = compare(ours, kata, pol)
    assert not a["agree"] and a["our_top"] == e5 and a["kata_top"] == d4
    assert abs(a["winrate_loss"] - 0.1) < 1e-9 and abs(a["points_loss"] - 1.5) < 1e-9
    assert a["our_rank_of_kata_top"] == 3 and abs(a["our_prior_on_kata_top"] - 0.1) < 1e-9
    assert a["our_share_on_kata_top"] == 0.2 and a["our_winrate_on_kata_top"] == 0.5
    same = compare({"candidates": [ours["candidates"][1]]}, kata, pol)
    assert same["agree"] and same["winrate_loss"] == 0.0
    assert compare(None, kata, pol) is None
