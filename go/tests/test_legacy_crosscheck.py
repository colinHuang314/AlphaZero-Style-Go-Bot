"""Compare the new engine with the original project's GoRules on random games.

Expected relationship:
  * captures and scoring match exactly
  * new-engine legal moves are a subset of old-engine legal moves; the only
    extra restriction is positional superko (the old engine only checked simple ko)
"""
import numpy as np

from gozero.go.state import GoState
from gozero.legacy.old_rules import Game as OldGame


def test_random_games_match_legacy_engine():
    rng = np.random.default_rng(7)
    superko_only = 0
    positions = 0
    for n in (5, 9):
        for _ in range(30 if n == 5 else 10):
            s = GoState.new(n)
            prev_old = None  # old engine's ko reference: board before opponent's last move
            while not s.is_terminal():
                board2d = s.board.reshape(n, n).astype(np.int8).copy()
                old_mask = OldGame.get_legal_moves_mask(board2d.copy(), s.to_play, prev_old, n)
                new_mask = s.legal_mask()
                positions += 1

                extra_new = (new_mask == 1) & (old_mask == 0)
                assert not extra_new.any(), f"new engine allows moves old engine forbids:\n{s}"
                for p in np.flatnonzero((old_mask == 1) & (new_mask == 0)):
                    # must be a superko ban: resulting position already occurred
                    child = OldGame.apply_move(board2d, divmod(int(p), n), s.to_play, n)
                    from gozero.go import engine
                    h = engine.board_hash(child.reshape(-1).astype(np.int8), s.rules.zob)
                    assert h in set(s.hist[:s.nhist].tolist())
                    superko_only += 1

                legal = np.flatnonzero(new_mask[:-1])
                if len(legal) == 0 or rng.random() < 0.03:
                    move = n * n
                else:
                    move = int(rng.choice(legal))
                s2 = s.play(move)
                if move != n * n:
                    old_after = OldGame.apply_move(board2d, divmod(move, n), s.to_play, n)
                    assert np.array_equal(old_after.reshape(-1), s2.board)
                prev_old = board2d
                s = s2

            old_score, _ = OldGame.final_score_tromp_taylor(s.board.reshape(n, n), n)
            assert abs(old_score - s.score()) < 1e-9
    print(f"checked {positions} positions, {superko_only} superko-only bans")
