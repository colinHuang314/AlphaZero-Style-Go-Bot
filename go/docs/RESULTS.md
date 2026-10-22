# go-zero: results

Four weeks (Sep 24 – Oct 22, 2026) of AlphaZero-style training on one laptop GPU (RTX 4060 Laptop, 8 GB),
from a rewrite of the original [AlphaZero-Style-Go-Bot](https://github.com/colinHuang314/AlphaZero-Style-Go-Bot)
to a 9×9 bot rated **≈2854 on CGOS** after 161 rated games. Pure zero: every training position comes from
self-play. KataGo and the CGOS bots are only opponents for *measuring* strength.

![Rating over time](rating_history.svg)

## Headline numbers

| | |
|---|---|
| Final CGOS 9×9 rating | **≈2854** as `gz-c2600-4060-tm85`, 161 games (Oct 21–22); the earlier 10-block bot was 2630 |
| vs bots rated 2900+ on CGOS | 32.5 / 79 (41%); earlier 10-block bot: 4 / 34 (12%) |
| vs GNU Go 3.8 (level 10) | 100–0 locally (c392, Sep 29); on CGOS E went 36–0 vs bots rated under 2200 |
| vs the original project's best net | 395–5 after the first night of training (equal 400-visit search) |
| Self-play | 504,065 games, 13.05M training positions, ~430 GPU-hours, 2,600 training cycles |
| Final net | 20 residual blocks × 128 channels, 5.95M parameters (policy, value, score, ownership heads) |
| Speed | ~250 → ~12,000 network evaluations/s in self-play (≈48×); ~5,500 visits/s for one game |
| Code | ~6,500 lines of Python (+ an 800-line browser UI), 64 unit/integration tests |

## Timeline

| Dates | Phase | Net | What happened |
|---|---|---|---|
| Sep 24 | Rewrite | — | numba rules engine, batched MCTS, ResNet with four heads, self-play/replay/trainer, continuous training loop, tests |
| Sep 24–30 | Nights 1–6 | 10×128 | lr 0.02 → 0.006 → 0.002 and 300 → 600 → 1000 visits for training targets; c79 → c439 |
| Oct 1 | Night 7 | 10×128 | c471 vs c439: 100–100. First plateau |
| Oct 2 | CGOS #1 | c439 | 71 games, ≈2630. Lost mostly to 2900+ bots; 8 draws as White exposed a komi bug |
| Oct 2–5 | Runs A, B, C | 10×128 | A: value target blended with search Q (+45). B: lower lr (+0). C: A's change continued (+10). The 10-block net was at its limit |
| Oct 5–15 | Run D | 15×128 | Grown from C by adding identity-initialised blocks. D_c1894 vs C_c611: **159–41 (+235)** |
| Oct 7–8 | Engine | — | CUDA graphs, overlapped GPU/CPU work, 2 self-play processes: ~3× faster self-play |
| Oct 13–15 | Analysis | — | 377 lost games re-checked by KataGo; visit-scaling test at 3,200 visits |
| Oct 15–20 | Run E | 20×128 | Grown from D. lr drop on day 3. E_c2600 vs D_c1894: **119–81 (+67)**; vs C_c611: 169–31 (+295) |
| Oct 20 | Rehearsal | E_c2600 | 40 games under CGOS conditions (komi 7, 5 min): no time losses, pondering reuses 74% of visits |
| Oct 21–22 | CGOS #2 | E_c2600 | 161 games, **≈2854** |

## How strength was measured

Self-play Elo against your own ancestors overstates real gains, so four separate measurements were used:

1. **Matches between checkpoints** (`tools/match.py`): 200 games at 400 visits, paired openings with colours
   swapped, Wilson 95% intervals. Used for every training decision.
2. **A KataGo yardstick** (`tools/yardstick.py`): our net at fixed visits vs KataGo at 1 visit (its raw
   policy), 200 games, komi 7.5. It is anchored to CGOS through c439's measured rating (c439 = 2630 ⇒ KataGo@1 ≈ 2871
   at 800 visits). A 100-game run is only ±80 Elo (the same net once scored 33 and then 23), so from Oct 10 every
   yardstick was 200 games.
3. **CGOS** (Computer Go Server, 9×9, komi 7, 5 minutes sudden death): the public rating.
4. **Life-and-death puzzles** (`problems/`, 29 positions): how many visits each net needs to solve them.

## Training runs

### Nights 1–7 (10 blocks × 128)

| Night | Change | Result (200 games at 400 visits unless noted) |
|---|---|---|
| 1 (c1–79) | baseline, lr 0.02 | c79 vs original `AZNET9_epoch_300`: **395–5** |
| 2 (c79–130) | lr 0.02 → 0.006 | one-time jump to ~75% vs c79 (+190) within 10 cycles, then flat |
| 2b (c130–162) | target visits 300 → 600 | c162 vs c79 74.5% (+186); c162 vs c130 61.5% (+81) |
| 3 (c162–244) | none | c244 vs c162 75.5% (+196), ~2.4 Elo per cycle |
| 4 (c244–341) | none | c341 vs c244 59.5% (+67), ~0.7 Elo per cycle (slowing) |
| 5 (c341–392) | target visits 600 → 1000 | c392 vs c341 53.5% (+24); c392 vs GNU Go **100–0** |
| 6 (c392–439) | lr 0.006 → 0.002 | c439 vs c392 61.5% (+81) |
| 7 (c439–471) | none | c471 vs c439 50.0%: plateau |

Lesson: on a fixed net, each lr drop or deeper search target gives one step of +80 to +190 Elo, then a plateau.

### Runs A–C: is the 10-block net the limit?

| Run | Change | vs c439 |
|---|---|---|
| A | value target = 0.5 · game result + 0.5 · root search Q | 113–87 (+45) |
| B | lr 0.002 → 0.0006 | 100–100 (+0) |
| C | A continued | C_c611 103–97 (+10); C vs A 100–100 |

### Runs D and E: growing the network

`tools/deepen.py` adds residual blocks whose last batch-norm scale and bias start at zero, so each new block
is an identity and the grown net plays exactly like the old one (verified: max output difference 0). Training
then continues from the same replay buffer.

| Net | Blocks | vs C_c611 (400 visits, 200 games) |
|---|---|---|
| D_c656 | 15 | 107–93 (+24) |
| D_c748 | 15 | 126–74 (+92) |
| D_c1736 | 15 | 160–40 (+241) |
| D_c1894 | 15 | 159–41 (+235) |
| **E_c2600** | **20** | **169–31 (+295)**; vs D_c1894 119–81 (+67, 95% CI +18 to +116) |

Run E dipped about 40 Elo below D for its first ~100 cycles (new blocks at lr 0.002), climbed back to level
with D, and passed it only after the lr drop to 0.0006.

## KataGo yardstick (our net at 800 visits vs KataGo@1, 200 games)

| Net | Score | CGOS-scale estimate |
|---|---|---|
| c439 | 20/100 | 2630 (anchor) |
| D_c1120 | 72/200 | ≈2771 |
| D_c1736 | 83/200 | ≈2811 |
| E_c2310 | 83/200 | ≈2811 |
| E_c2470 | 81/200 | ≈2804 |

At 3,200 visits: c439 80/200, D_c1736 114/200, E_c2600 108/200. Deeper search shrinks the gaps between nets
(c439 gains +141 Elo from 4× visits, D only +91), so the 800-visit numbers overstate later nets slightly.

## Why the bot loses (377 lost yardstick games re-checked by KataGo)

- 97 losses came from the random two-move openings. In the other 280, the game turned at a median of
  **move 20** with a median swing of only 3.5 points: positional misjudgments early in the middlegame, not
  late blunders.
- Right after the losing move our net still thought it had about 55%; KataGo said 11%. In 177 of 280 cases our
  net valued its losing move at least as highly as KataGo's choice.
- KataGo's move had a median prior of 5.7% in our policy (below 2% in 85 cases).
- More search helps: 800 visits repeated the losing move 178/280 times; 6,400 visits avoided it 198/280.

## CGOS

| | c439 (Oct 2) | E_c2600 (Oct 21–22) |
|---|---|---|
| Net | 10 blocks, 3.0M parameters | 20 blocks, 5.95M parameters |
| Engine | time management | + pondering, komi correction, resignation |
| Games | 71 | 161 |
| Rating | 2630 | **≈2854** |
| vs 2900+ | 4/34 | 32.5/79 |
| vs 2600–2900 | 4/18 | 16.5/46 |
| vs under 2200 | 16/17 | 36/36 |
| Losses on time | 0 | 0 |
| Clock used (of 300 s) | not tracked | mean 37%, worst 81% |

The engine on CGOS averaged ~20,000 visits per move (up to the 50,000 cap): pondering on the opponent's
time reused ~70% of each move's search.

**Known limitation: draws as White.** 34 of E's 81 White games were draws (Black: 4 of 80). The net was trained at
komi 7.5, where a half-point win for White is a win; CGOS uses 7.0, where it is a draw. In those games the search
rated White's position +0.7 to +0.95 by move 60 and then settled for the half point. `--komi-fix` (which
moves the probability of that exact result to a draw) removed most such draws in local tests, but not against
strong CGOS opponents. A score-based utility, as KataGo uses, is the natural next fix.

## Engine features (GTP)

| Feature | Effect (local tests) |
|---|---|
| Time management (`--time-manage --tm-use 0.85`) | +241 Elo vs a fixed 3,200 visits at the same 5-minute clock |
| Pondering (`--ponder`) | ~1.65× visits per move in timed games |
| Komi correction (`--komi-fix`) | +38 Elo at komi 7 (White draws 27 → 5 in self-play) |
| Resignation (`--resign -0.97`, 3 moves in a row) | no rating effect; ends lost games cleanly |
| Symmetry pruning, FPU reduction, tree reuse | faster, stronger search on symmetric boards |

## Speed work

| Step | Self-play evaluations/s |
|---|---|
| Original project (one position per GPU call) | ~250 |
| Rewrite: 256 games batched, numba rules engine | ~3,500 |
| CUDA graphs + GPU work overlapped with tree search | ~7,200 |
| 2 self-play worker processes | ~12,000–13,400 |

## What I would do next

1. A score-based utility in the engine (fixes the White draws; ~+20–40 CGOS Elo estimated).
2. KataGo's training refinements that stay pure zero: global pooling in the trunk, policy-surprise weighting,
   forced playouts with policy-target pruning.
3. A wider net (192–256 channels) rather than a deeper one, plus randomised komi in self-play.

The full night-by-night log is in [NOTES.md](../NOTES.md) and the day-by-day plan in [PLAN.md](../PLAN.md).
