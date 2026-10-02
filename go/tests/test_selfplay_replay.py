import numpy as np

from gozero.go import symmetry
from gozero.go.features import make_features
from gozero.mcts.evaluator import UniformEvaluator
from gozero.mcts.mcts import MCTSConfig
from gozero.selfplay.selfplay import SelfPlay, SelfPlayConfig, record_to_training
from gozero.train.replay import ReplayBuffer, window_size


def play_games(n=5, count=4):
    cfg = SelfPlayConfig(board_size=n, num_parallel=count, full_visits=16, fast_visits=4, full_prob=0.5,
                         mcts=MCTSConfig())
    sp = SelfPlay(cfg, UniformEvaluator(), seed=0)
    done = []
    while len(done) < count:
        done += sp.step()
    return done


def test_training_targets_perspective():
    for rec in play_games():
        d = record_to_training(rec, 5)
        if d is None:
            continue
        for i, (_, pi, to_play, _, q) in enumerate(rec.samples):
            assert d["z"][i] == rec.winner * to_play
            assert abs(d["q"][i] - q) < 1e-6 and -1 <= q <= 1   # root value, side to move's view
            # ownership target: +1 = owned by the side to move at that position
            assert np.array_equal(d["ownership"][i], rec.final_ownership * to_play)
            assert abs(d["score"][i] - rec.score * to_play) < 1e-5
            assert abs(pi.sum() - 1) < 1e-5
        # final ownership is consistent with the final score (Tromp-Taylor)
        if not rec.resigned:
            assert abs(rec.final_ownership.sum() - 3.5 - rec.score) < 1e-6 or \
                abs(rec.final_ownership.sum() - rec.score - 3.5) < 1e-6


def test_policy_targets_are_visit_distributions_not_sharpened():
    for rec in play_games():
        for f, pi, *_ in rec.samples:
            empty = (f[0] == 0) & (f[1] == 0)
            n_legal = int((empty & (f[8] == 0)).sum()) + 1
            if n_legal < 5:
                continue  # e.g. only pass legal -> one-hot is correct
            # with 16 visits over >=5 moves at uniform priors, a one-hot target would mean temperature leaked in
            assert (pi > 0).sum() > 1


def test_replay_roundtrip_and_augmentation(tmp_path):
    n = 5
    buf = ReplayBuffer(100, n)
    recs = play_games()
    for rec in recs:
        d = record_to_training(rec, n)
        if d is not None:
            buf.add(d)
    assert buf.size > 0
    # unaugmented gather returns exactly what was stored
    idx = buf.recent_indices(buf.size)
    g = buf.gather(idx)
    flat = [s for rec in recs for s in rec.samples][-buf.size:]
    stored = {tuple(np.packbits(f.reshape(-1))) for f, *_ in flat}
    for f in g["features"]:
        assert tuple(np.packbits(f.reshape(-1))) in stored
    # augmented batch: each sample must equal some symmetry of a stored sample, with pi transformed the same way
    rng = np.random.default_rng(0)
    b = buf.sample(32, buf.size, rng)
    base = buf.gather(idx)
    for f, pi in zip(b["features"], b["pi"]):
        ok = False
        for j in range(len(idx)):
            for k in range(8):
                if np.array_equal(symmetry.transform_planes(base["features"][j], k), f) and \
                        np.allclose(symmetry.transform_policy(base["pi"][j], k, n), pi):
                    ok = True
                    break
            if ok:
                break
        assert ok
    # save / load
    p = str(tmp_path / "b.npz")
    buf.save(p)
    b2 = ReplayBuffer.load(p)
    assert b2.size == buf.size and b2.ptr == buf.ptr
    assert np.array_equal(b2.gather(idx)["features"], g["features"])
    assert np.allclose(b2.gather(idx)["q"], g["q"], atol=1e-3)
    # a buffer saved before root q was recorded loads with q = z (the old value target)
    old = {k: v for k, v in np.load(p).items() if k != "q"}
    p_old = str(tmp_path / "old.npz")
    np.savez(p_old, **old)
    assert np.array_equal(ReplayBuffer.load(p_old).gather(idx)["q"], g["z"])


def test_value_target_blend():
    import torch
    from gozero.train.trainer import TrainConfig, compute_losses

    import math

    class Fixed(torch.nn.Module):  # constant outputs (value logit 1), so the losses depend only on the targets
        def forward(self, x):
            b = x.shape[0]
            return {"policy": torch.zeros(b, 26), "value_logit": torch.ones(b), "ownership": torch.zeros(b, 25),
                    "score": torch.zeros(b)}

    t = {"features": torch.zeros(1, 1), "pi": torch.full((1, 26), 1 / 26), "z": torch.tensor([1.0]),
         "q": torch.tensor([0.2]), "ownership": torch.zeros(1, 25), "score": torch.zeros(1),
         "aux_weight": torch.ones(1)}
    sig = 1 / (1 + math.exp(-1))
    bce = lambda y: -(y * math.log(sig) + (1 - y) * math.log(1 - sig))
    _, s0 = compute_losses(Fixed(), t, TrainConfig(value_q_weight=0.0))
    _, s5 = compute_losses(Fixed(), t, TrainConfig(value_q_weight=0.5))
    assert abs(s0["value_target"] - bce(1.0)) < 1e-5       # w = 0: target is the game result (z = +1 -> 1)
    assert abs(s5["value_target"] - bce(0.8)) < 1e-5       # w = 0.5: 0.5 * 1 + 0.5 * 0.2 = 0.6 -> 0.8
    assert abs(s0["value"] - s5["value"]) < 1e-6           # `value` always measures against the game result


def test_ring_wraparound_keeps_newest():
    n = 5
    buf = ReplayBuffer(10, n)
    for i in range(3):
        d = {"features": np.zeros((6, 16, n, n), np.uint8) + (i % 2), "pi": np.full((6, 26), 1 / 26, np.float32),
             "z": np.full(6, i, np.int8), "ownership": np.zeros((6, 25), np.int8),
             "score": np.zeros(6, np.float32), "aux_weight": np.ones(6, np.float32)}
        buf.add(d)
    assert buf.size == 10 and buf.total_added == 18
    assert sorted(buf.gather(buf.recent_indices(6))["z"].tolist()) == [2] * 6
    assert sorted(buf.gather(buf.recent_indices(6))["q"].tolist()) == [2] * 6  # no q given -> q = z


def test_window_growth():
    assert window_size(1000, 25000, 400000) == 25000
    w1 = window_size(100000, 25000, 400000)
    w2 = window_size(1000000, 25000, 400000)
    assert 25000 < w1 < w2 <= 400000
