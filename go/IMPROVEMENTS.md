# How go-zero improves on AlphaZero-Style-Go-Bot

**Outcome.** A network trained from scratch for one night (9.7 h, 16,188 self-play
games) beats the original project's best 9×9 model in **98.8%** of games
(395–5 vs `AZNET9_epoch_300`; 397–3 vs `AZNET9_epoch_100`). Both sides got equal
search (400 visits per move) with the same search code, colors were swapped on
every random opening, and 400 games were played per opponent. When the old model
got its own preferred search setting (`c_puct=3`), the result was the same:
196–4. For comparison, the original models trained for about 2 weeks.

## Summary, largest impact first

This ranking is my judgment of how much each change contributed to that result.
It is **not** the output of ablations: I didn't retrain with each fix removed,
because each retrain costs a night. Where a number exists it is measured, and
the evidence column says which.

| # | Change | Old behavior | New behavior | Evidence |
|---|---|---|---|---|
| 1 | **Policy targets = raw visit counts** | Targets were stored *after* temperature, so from move 19 on every target was one-hot | The visit distribution is stored unchanged; temperature only affects which move is played | Code review; test `test_policy_targets_are_visit_distributions_not_sharpened` |
| 2 | **25–70× faster self-play** | One position per GPU call, Python flood fill per legality check, about 250 evals/s | 256 games searched together, numba rules engine, CPU and GPU work overlapped: 4–17k evals/s | Benchmarked. About 1,700 games/hour against about 20/hour |
| 3 | **No more value-head memorization** | About 30 games per generation, 8 stored augmentations, 4+ passes: each position seen ~40×. Value loss fell to 0.16 in one step | Continuous replay window, symmetry drawn at random per sample, ~4 samples trained per new sample, held-out validation games | Value swing across board symmetries: **0.30 → 0.055**. Validation−training value gap 0.02–0.04 |
| 4 | **Search bugs fixed** | Dirichlet noise never applied; root saw no board history; an opponent's pass before the root was ignored; `MIN_PRIOR` floor flattened every prior | All four fixed, each with a unit test | Old model: extra search made tactics *worse* (2/4 → 1/4). New model: 3/4 at every search size |
| 5 | **Continuous training instead of gating** | New nets had to win a 16-game arena at 53% to generate data; noisy promotions, stalled data | Always self-play with the latest net (AlphaZero); strength tracked against fixed anchors with confidence intervals | 85–100% vs old300 from 1.3 h on |
| 6 | **Extra training signal: ownership + score heads** | Only a win/loss target per game | Also predicts final ownership of every point and the score margin (KataGo) | Design choice from KataGo's paper; not ablated here |
| 7 | **Better input features** | 8 boards of absolute black/white history | Stones relative to the side to move, liberty counts, illegal points, last 4 moves, "opponent just passed" | Removes the root/training history mismatch; not ablated |
| 8 | **Smarter use of search budget** | Fixed time per move, time-based and hardware-dependent | Fixed visits; playout cap randomization (25% full searches for training targets, 75% cheap); tree reuse between moves; smooth temperature decay | About 2.5× more games for the same compute (120 vs 300 average visits per move) |
| 9 | **Correct rules** | Simple ko only; rules rebuilt each call | Positional superko (Tromp-Taylor), incremental hashing | Cross-checked against the old engine on 3,885 positions: identical captures and scores |
| 10 | **Engineering** | Hard-coded paths, manual epoch files, no tests | One config file per run; safe Ctrl+C and resume; 31 tests; `match.py` / `diagnose.py`; browser UI | — |

---

## Details

### 1. Policy targets were one-hot for most of each game (the biggest bug)

In `TrainingLoop.py:298-309` the loop did this:

```python
pi = apply_temperature(pi, temp)            # temp = 0 after move 18 -> one-hot argmax
...
data_history.append((state_planes, pi, to_play))   # this pi becomes the training target
```

With `TEMPERATURE_SCHEDULE = [..., (0, 20), (0, 9999)]`, every position after
move 18 or 20 had a one-hot target: "the most-visited move gets probability 1,
everything else 0". Two things go wrong:

- **Most of what the search found is thrown away.** A visit split of 40/35/25
  between three moves says they are nearly equal. A one-hot target says two of
  them are worthless.
- **The policy becomes overconfident.** A net trained on one-hot targets learns
  near-deterministic priors, and with priors that confident the search rarely
  explores alternatives. The original README's "overfitting" note, and the
  `MIN_PRIOR=0.01` floor added to fight it, are consistent with this.

**Fix** (`gozero/selfplay/selfplay.py`): record `root.visit_distribution()`
before choosing a move. Temperature only affects which move is *played*, as in
AlphaZero.

### 2. Throughput: from ~250 to 4,000–17,000 evaluations per second

Compute was the README's first suspected bottleneck, and it was real. The
original spent most of its time on overhead, not on the GPU:

- **One position per GPU call.** Every simulation ran the network on a batch of
  one. go-zero runs 256 games at once and evaluates all their new leaf
  positions in one batch (`gozero/mcts/mcts.py`, `BatchedMCTS`).
- **Legality checks in Python.** `get_legal_moves_mask` placed a stone and
  flood-filled neighbor groups for every empty point, at every node. The new
  engine (`gozero/go/engine.py`) is compiled with numba. It labels all groups
  and their liberties once per position and derives every move's legality from
  that. About 22× faster.
- **GPU and CPU took turns.** The games are split into two halves, and the CPU
  walks one half's trees while the GPU evaluates the other half.
- **Search selection** (the PUCT formula) moved to numba as well. It was the
  top entry in the profile.

| Setup | Evals/s |
|---|---|
| Original (per README comments) | ~150–250 |
| go-zero, 10×128 net (the one trained) | 4,000–7,500 |
| go-zero, 6×64 net | ~17,000 |

The result: 16,188 games in one night, where the original project had about
4,000 games in total.

### 3. Value-head memorization and the data pipeline

The original's own example log shows the problem. In the first training step,
value loss fell from 1.00 to 0.16. A value head can't learn Go that fast; it
was memorizing which game each position came from. The causes:

- **Tiny data, reused heavily.** `SELFPLAY_GAMES_PER_ITER = 6` × `ARENA_INTERVAL = 5`
  gives about 30 games per generation. All 8 symmetries were stored as separate
  samples, and training ran 4 passes over the buffer. Each real position was
  seen about 40 times.
- **Correlated samples.** Every position in a game shares the same outcome, so
  memorizing the game is the easiest way to fit the value target.

**Measured effect.** A net that understands positions gives nearly the same value
for a board and its rotations or mirror images. `tools/diagnose.py` checks this
on positions from each model's own play:

| | Old epoch 300 | New cycle 79 |
|---|---|---|
| Value std across the 8 symmetries | 0.30–0.40 | **0.055** |
| Top move agreement across symmetries | 59–68% | 69% |
| Policy L1 difference across symmetries | 0.46–0.47 | **0.24** |

**Fix** (`gozero/train/replay.py`, `gozero/loop.py`):
- A ring buffer stores each position once. A random symmetry is applied each
  time it is sampled, so augmentation adds variety instead of copies.
- The buffer window grows with the data (KataGo's schedule), so early low-quality
  games age out.
- About 4 samples are trained per new sample, set by `train_ratio`.
- 5% of games are held out and never trained on. Validation loss is logged
  every 5 cycles, so overfitting is visible, not guessed.

The held-out value predictions are also reasonably calibrated. Positions the
model gives under 10% win win 4% of the time, and positions above 90% win 98% of
the time.

### 4. Search bugs

| Bug | Location | Effect | Fix |
|---|---|---|---|
| Dirichlet noise never applied | `MCTS_Go.py:155,175`: `sim` starts at 1, noise needs `sim == 0` | No root exploration in self-play, so blind spots were never tested | Noise applied at every full-search root; tested by `test_dirichlet_noise_applied_at_root_only_when_requested` |
| Root had no board history | `MCTS_Go.py:152` passes `board_history=None` | During search the net saw 7 empty history planes, unlike in training | New features don't need history; the old net's adapter rebuilds its full 8-board history |
| Opponent's pass before the root ignored | `MCTS_Go.py:160-170` counts passes only inside the tree | After the opponent passed, passing (which ends the game) was valued as if play continued | `GoState.passes` carries the real count; tested both ways (pass when ahead, don't pass when behind) |
| `MIN_PRIOR = 0.01` floor on every legal move | `MCTS_Go.py:110` | With ~80 legal moves this adds ~0.8 probability before renormalizing, so a 0.9 prior became ~0.5 | Plain masked softmax |

The tactics suite in `tools/diagnose.py` shows the combined effect of these
bugs and the weak value head:

| Puzzle | Old epoch 300 (policy / 32 visits / 200 visits) | New cycle 79 |
|---|---|---|
| Capture in 1 | ✓ / ✗ / ✗ | ✓ / ✓ / ✓ |
| Escape atari | ✗ / ✗ / ✗ | ✓ / ✓ / ✓ |
| Capture 4 stones | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ |
| Save group with 1 liberty | ✗ / ✗ / ✗ | ✗ / ✗ / ✗ |

For the old model, *more search made it worse*: the raw policy found the
capture, and the search talked itself out of it. That points to a misleading
value head (section 3). The new model's answers hold as search grows. The last
puzzle is a group that is probably dead even after the "saving" move, so
playing elsewhere may be a reasonable judgment, not a miss.

### 5. Continuous training instead of arena gating

The original followed AlphaGo Zero: a new net had to beat the champion in an arena
before its games were used. With `ARENA_GAMES = 16`, a 53% threshold and early
stopping, a net *no stronger than the champion* got promoted a good share of the
time. The README noticed this ("beating the current champion does not guarantee
real improvement"). The arena also cost about as much GPU time as self-play.

go-zero follows AlphaZero: self-play always uses the latest net. Progress is
measured instead of used as a gate. Every 10 cycles (about 70 minutes), the loop
plays 64 games against the fixed anchors and against the previous snapshot.
Results are reported with Wilson confidence intervals and Elo, and the final
evaluation is a separate, larger match (`tools/match.py`, protocol in `NOTES.md`).

### 6. Ownership and score heads

A game gives one win/loss bit for about 100 positions. The ownership head
predicts who owns each of the 81 points at game end, and the score head predicts
the final margin. That gives the shared network far more learning signal per
game. The KataGo paper reports these auxiliary targets as a clear gain in sample
efficiency. The UI also uses the heads: the ownership overlay, the dead-stone markers, and the
score lead.

### 7. Input features

The old input was 8 past boards in absolute colors plus a side-to-move plane.
The new input has 16 planes:
- stones as "mine" and "opponent's", so the same weights serve both colors
- liberty counts (1 / 2 / 3 or more) for both sides
- empty points that are illegal to play
- the last 4 moves
- whether the opponent just passed
- whose turn it is (because komi makes the colors asymmetric)
- a constant plane so the net can find the board edge

These come straight from the rules, not from human games, so training is still
from scratch. They also remove the history mismatch in section 4.

### 8. Search budget

- **Fixed visits, not fixed time.** The old search time depended on hardware
  speed, which made data quality depend on what else the machine was doing.
- **Playout cap randomization** (KataGo). 25% of moves get a full search
  (300 visits) and become policy training samples. The rest get a cheap 60-visit
  search, just to keep the game going with a reasonable move. The average falls
  from 300 to 120 visits per move, so the same compute plays about 2.5× more
  games, and every game adds win/loss, ownership and score targets. Policy
  targets still come only from full searches.
- **Tree reuse.** The subtree under the chosen move is kept for the next move.
- **Smooth temperature decay** (1.0 → 0.25 with an 8-move half-life), replacing
  the step schedule. This affects only which move is played, never the targets.

### 9. Rules engine

- **Positional superko** (Tromp-Taylor): a move may not recreate *any* earlier
  position. Previously only an immediate retake was blocked. Zobrist hashes
  are updated with each move, so the check is cheap.
- **Cross-checked against the old engine** on random games: captures and
  scores match exactly on all 3,885 positions. The only disagreements are 3
  moves the new engine bans under superko (`tests/test_legacy_crosscheck.py`).

### 10. Engineering and tooling

- **Configuration.** One YAML file per run (`configs/9x9.yaml`), and the config
  used is saved with the run. No hard-coded paths or hand-made `curr_epoch.txt` /
  `champion.txt` files.
- **Resuming.** Ctrl+C or `--hours N` saves the model, optimizer and replay
  buffer, and the next run resumes from them. Files are written atomically.
  Overnight runs were designed around this.
- **31 tests** covering captures, suicide, ko and superko, scoring, hashing,
  symmetries, features, MCTS accounting and pass logic, virtual loss,
  self-play targets, replay augmentation and the old-model adapter.
- **`tools/match.py`**: the official evaluation, with equal visits, paired
  color-swapped openings and confidence intervals.
- **`tools/diagnose.py`**: checks each part of a model separately (symmetry
  consistency, held-out accuracy and calibration, tactics).
- **`NOTES.md`**: which metric points to which hyperparameter change.
- **Browser UI** (`python -m gozero.ui.server`). The same visual style as the
  original, plus a win-rate graph, a top-moves table with principal variations,
  an ownership overlay, bot-vs-bot, SGF load/save, and live search at about 10×
  the speed.

## What was kept

- The overall AlphaZero recipe: self-play with a ResNet policy/value net and PUCT search.
- Board size 9×9 and komi 7.5, so results are directly comparable.
- Tromp-Taylor area scoring.
- The original checkpoints, as fixed opponents in `anchors/`. An adapter
  rebuilds their exact 17-plane input encoding, verified plane-for-plane
  against the original `make_input_planes`.
- The UI's visual language.

## Caveats

- **No ablations**, as noted at the top. Retraining with each fix removed would
  cost a night each, so the ranking above is judgment backed by the listed
  measurements.
- **The old models ran inside the new search**, as agreed for the evaluation.
  They got their true board history and the new engine's legal-move mask, with
  no `MIN_PRIOR` floor. They were *not* run through the original `MCTS_Go.py`
  with its bugs. That setup would likely make them weaker still.
- **All 12 losses in the 1,000 evaluation games came with the new model as
  Black**, so it may still be slightly weaker as Black at 7.5 komi.
