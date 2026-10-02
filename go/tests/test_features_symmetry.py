import numpy as np

from gozero.go import symmetry
from gozero.go.features import NUM_PLANES, make_features
from gozero.go.state import GoState


def random_state(n, moves, seed):
    rng = np.random.default_rng(seed)
    s = GoState.new(n)
    for _ in range(moves):
        legal = np.flatnonzero(s.legal_mask()[:-1])
        s = s.play(int(rng.choice(legal)))
    return s


def test_symmetry_group():
    n = 7
    perms = symmetry.permutations(n)
    assert len({tuple(p) for p in perms}) == 8
    x = np.arange(n * n)
    for k in range(8):
        y = symmetry.transform_flat(x, k, n)
        assert np.array_equal(symmetry.transform_flat(y, symmetry.inverse(k), n), x)


def test_planes_and_flat_agree():
    n = 9
    a = np.random.default_rng(0).random((n, n))
    for k in range(8):
        via_planes = symmetry.transform_planes(a, k).reshape(-1)
        via_flat = symmetry.transform_flat(a.reshape(-1), k, n)
        assert np.array_equal(via_planes, via_flat)


def test_features_equivariant():
    """Features of a transformed position == transformed features of the position."""
    n = 9
    s = random_state(n, 40, seed=3)
    f = make_features(s)
    for k in range(8):
        tb = symmetry.transform_flat(s.board, k, n)
        t = GoState(s.rules, tb.copy(), s.to_play)
        t.last_moves = tuple(int(symmetry.permutations(n)[symmetry.inverse(k)][m]) if 0 <= m < n * n else m
                             for m in s.last_moves)
        # superko history is not symmetric in general; compare only when it doesn't matter
        f_t = make_features(t)
        expected = symmetry.transform_planes(f, k)
        assert np.array_equal(np.delete(f_t, 8, 0), np.delete(expected, 8, 0))


def test_feature_basics():
    s = GoState.new(5)
    f = make_features(s)
    assert f.shape == (NUM_PLANES, 5, 5)
    assert f[14].all() and f[15].all() and not f[13].any()
    s = s.play(12)  # black center
    f = make_features(s)  # white to move
    assert f[1, 2, 2] == 1 and f[0].sum() == 0  # black stone is "opponent"
    assert f[7, 2, 2] == 1  # 4 liberties -> >=3 plane
    assert f[9, 2, 2] == 1  # last move
    assert not f[14].any()
    s = s.play(25)  # white passes
    assert make_features(s)[13].all()
