import numpy as np

from gozero.go.state import GoState, str_to_move
from gozero.mcts.evaluator import RolloutEvaluator, UniformEvaluator
from gozero.mcts.mcts import BatchedMCTS, MCTSConfig, Node, pick_move


def position(n, black, white, to_play):
    s = GoState.new(n)
    b = s.board.copy()
    for m in black:
        b[str_to_move(m, n)] = 1
    for m in white:
        b[str_to_move(m, n)] = -1
    return GoState(s.rules, b, to_play)


def test_visit_accounting():
    mcts = BatchedMCTS(MCTSConfig(), np.random.default_rng(0))
    roots = [Node(GoState.new(5)) for _ in range(3)]
    mcts.search(roots, UniformEvaluator(), [10, 50, 100])
    for r, v in zip(roots, [10, 50, 100]):
        assert r.N == v
        assert r.Nc.sum() == v
        assert abs(r.visit_distribution(26).sum() - 1) < 1e-6


def test_tree_reuse_keeps_visits():
    mcts = BatchedMCTS(MCTSConfig(), np.random.default_rng(0))
    root = Node(GoState.new(5))
    mcts.search([root], UniformEvaluator(), 200)
    m = pick_move(root, 0, np.random.default_rng(0))
    child = root.child_by_move(m)
    before = child.N
    assert before > 0
    mcts.search([child], UniformEvaluator(), before + 20)
    assert child.N == before + 20


def test_dirichlet_noise_applied_at_root_only_when_requested():
    mcts = BatchedMCTS(MCTSConfig(dirichlet_eps=0.25), np.random.default_rng(0))
    a, b = Node(GoState.new(5)), Node(GoState.new(5))
    mcts.search([a, b], UniformEvaluator(), 5, noise=[True, False])
    assert not np.allclose(a.P, a.P_raw)
    assert np.allclose(b.P, b.P_raw)


def _after_opponent_pass(black, white):
    # white to move, then white passes so black sees "opponent just passed"
    s = position(5, black, white, to_play=-1)
    return s.play(s.rules.N)


def test_passes_to_win_when_ahead_after_opponent_pass():
    # Black walls off columns A-C (15 points) vs white D-E (10); opponent just passed -> passing wins now.
    s = _after_opponent_pass(black=["C1", "C2", "C3", "C4", "C5"], white=["D1", "D2", "D3", "D4", "D5"])
    assert s.to_play == 1 and s.passes == 1
    assert s.play(s.rules.N).score() > 0
    mcts = BatchedMCTS(MCTSConfig(), np.random.default_rng(0))
    root = Node(s)
    mcts.search([root], UniformEvaluator(), 200)
    assert pick_move(root, 0, np.random.default_rng(0)) == s.rules.N


def test_does_not_pass_when_behind_after_opponent_pass():
    # Black is behind on the board; passing now loses immediately.
    s = _after_opponent_pass(black=["B1", "B2", "B3", "B4", "B5"], white=["D1", "D2", "D3", "D4", "D5"])
    assert s.play(s.rules.N).score() < 0
    mcts = BatchedMCTS(MCTSConfig(), np.random.default_rng(0))
    root = Node(s)
    mcts.search([root], UniformEvaluator(), 200)
    assert pick_move(root, 0, np.random.default_rng(0)) != s.rules.N


def test_finds_capture_of_large_group():
    # White group C2-C3-C4 in atari (last liberty C5). Black to capture.
    s = position(5, black=["B2", "B3", "B4", "D2", "D3", "D4", "C1"], white=["C2", "C3", "C4", "E5", "A5"], to_play=1)
    mcts = BatchedMCTS(MCTSConfig(c_puct=1.0), np.random.default_rng(0))
    root = Node(s)
    mcts.search([root], RolloutEvaluator(rollouts=4, seed=0), 300)
    assert pick_move(root, 0, np.random.default_rng(0)) == str_to_move("C5", 5)


def test_arena_accounting_and_rollout_beats_uniform():
    from gozero.eval.arena import Player, play_match
    r = play_match(Player("rollout", RolloutEvaluator(rollouts=2, seed=0), 48),
                   Player("uniform", UniformEvaluator(), 48), num_pairs=4, n=5, seed=0)
    assert r.games_played == 8 and r.a_games_as_black == 4
    assert r.a_wins >= 6, r.summary()


def test_search_single_matches_visit_target_and_finds_capture():
    from gozero.mcts.mcts import principal_variation, search_single
    s = position(5, black=["B2", "B3", "B4", "D2", "D3", "D4", "C1"], white=["C2", "C3", "C4", "E5", "A5"], to_play=1)
    root = Node(s)
    search_single(root, RolloutEvaluator(rollouts=4, seed=0), MCTSConfig(c_puct=1.0), 300, leaf_batch=8)
    assert root.N == 300 and root.Nc.sum() == 300  # virtual loss fully removed
    assert pick_move(root, 0, np.random.default_rng(0)) == str_to_move("C5", 5)
    pv = principal_variation(root)
    assert pv[0] == str_to_move("C5", 5) and len(pv) >= 2
