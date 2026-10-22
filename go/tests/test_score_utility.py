import math

import numpy as np

from gozero.eval.arena import MatchResult, Player, play_match
from gozero.go.state import GoState
from gozero.mcts.evaluator import RolloutEvaluator, ScoreUtility, UniformEvaluator


def _states(komi):
    black = GoState.new(9, komi)
    white = black.play(black.rules.N)  # black passes: white to move
    return black, white


def test_no_change_at_training_komi():
    u = ScoreUtility(komi_fix=True)
    b, w = _states(7.5)
    v = np.array([0.98, -0.98])
    assert np.allclose(u(v, np.array([0.5, -0.5]), [w, b]), v)


def test_half_point_white_win_counts_as_likely_draw_at_komi_7():
    u = ScoreUtility(komi_fix=True, sigma=2.0)
    b, w = _states(7.0)
    p_bin = math.erf(1 / (2 * math.sqrt(2)))  # P(|x| < 1) for x ~ N(0, 2): the draw bin is 2 points wide
    # White to move, net says White +0.5 at 7.5 with value 0.98 (that is Black ahead by exactly 7: a draw at 7.0)
    vw = u(np.array([0.98]), np.array([0.5]), [w])[0]
    assert math.isclose(vw, 0.98 - p_bin, rel_tol=1e-9)
    # the same position from Black's side: its loss is partly a draw, so its value goes up by the same amount
    vb = u(np.array([-0.98]), np.array([-0.5]), [b])[0]
    assert math.isclose(vb, -0.98 + p_bin, rel_tol=1e-9)
    # a clear White win (by 10.5) is barely touched; and the shift never exceeds the win probability
    assert 0.98 - u(np.array([0.98]), np.array([10.5]), [w])[0] < 1e-3
    assert u(np.array([-1.0]), np.array([0.5]), [w])[0] == -1.0


def test_komi_8_moves_black_wins_to_draws():
    u = ScoreUtility(komi_fix=True)
    b, w = _states(8.0)
    assert u(np.array([0.9]), np.array([0.5]), [b])[0] < 0.9   # Black +0.5 at 7.5 is a draw at 8
    assert u(np.array([-0.9]), np.array([-0.5]), [w])[0] > -0.9


def test_score_bonus_prefers_bigger_margin_and_is_small():
    u = ScoreUtility(komi_fix=False, score_weight=0.05)
    b, w = _states(7.0)
    vals = u(np.full(4, 0.99), np.array([0.5, 2.5, 10.5, -30.0]), [w] * 4)
    assert vals[0] < vals[1] < vals[2]
    assert np.all(np.abs(vals - 0.99) <= 0.05)
    # the margin is taken at the real komi: White +0.5 at 7.5 is exactly 0 at 7.0, so no bonus
    assert math.isclose(vals[0], 0.99, abs_tol=1e-12)


def test_match_counts_draws_as_half():
    r = MatchResult("a", "b", a_wins=3, b_wins=1, draws=2, lengths=[1])
    assert r.games_played == 6 and math.isclose(r.a_winrate, 4 / 6)
    assert "2 draws" in r.summary() and ": 3-1 (" in r.summary()


def test_integer_komi_match_runs_and_accounts():
    a = Player("rollout", RolloutEvaluator(rollouts=1, seed=0), 16)
    b = Player("uniform", UniformEvaluator(), 16)
    r = play_match(a, b, num_pairs=4, n=5, seed=1, komi=3.0)
    assert r.games_played == 8 and r.a_wins + r.b_wins + r.draws == 8
    assert all(st == 0 for _, w, _, st in r.games if w == 0)
