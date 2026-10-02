import os

import numpy as np
import pytest
import torch

from gozero.go.state import GoState
from gozero.legacy import old_board_ops
from gozero.legacy.old_net import load_legacy
from gozero.mcts.evaluator import LegacyEvaluator

ANCHOR = os.path.join(os.path.dirname(__file__), "..", "anchors", "AZNET9_epoch_300.pt")


def test_legacy_features_match_original_encoding():
    rng = np.random.default_rng(0)
    n = 9
    s = GoState.new(n)
    history = [s.board.reshape(n, n).copy()]  # the old self-play loop's board_history
    ev = LegacyEvaluator.__new__(LegacyEvaluator)
    ev.n = n
    for _ in range(30):
        ours = ev.features([s])[0]
        theirs = old_board_ops.make_input_planes(history[-8:], s.to_play, n).astype(np.uint8)
        assert np.array_equal(ours, theirs)
        legal = np.flatnonzero(s.legal_mask()[:-1])
        move = n * n if rng.random() < 0.1 else int(rng.choice(legal))
        s = s.play(move)
        history.append(s.board.reshape(n, n).copy())


@pytest.mark.skipif(not os.path.exists(ANCHOR), reason="anchor checkpoint missing")
def test_legacy_symmetry_plumbing():
    """A position that is itself symmetric must give identical outputs under every symmetry.

    (How consistent the old net is on asymmetric positions is a model-quality question,
    measured by tools/diagnose.py, not a correctness test.)
    """
    net = load_legacy(ANCHOR)
    ev = LegacyEvaluator(net, device=torch.device("cpu"), use_amp=False)
    s = GoState.new(9).play(40)  # tengen
    vals = [ev.raw([s], k)["value"][0] for k in range(8)]
    assert np.allclose(vals, vals[0], atol=1e-5)
