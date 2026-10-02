# go-zero

A rebuilt AlphaZero-style Go engine for 9×9 (and smaller boards), based on
[AlphaZero-Style-Go-Bot](https://github.com/colinHuang314/AlphaZero-Style-Go-Bot).
It is fully standalone: the original checkpoints are copied into `anchors/` and
serve only as fixed opponents for measuring progress.

**Goal:** a model trained from scratch that beats the original 9×9 models
(`AZNET9_epoch_100`, `AZNET9_epoch_300`) in ≥95% of games at equal search visits.

## Quick start

```
pip install -r requirements.txt
python -m pytest                      # rules, features, MCTS, self-play, replay tests
python -m gozero.loop --config configs/9x9.yaml --run runs/9x9_a --hours 9
```
Stop anytime with Ctrl+C. Rerunning the same command resumes from where it left off.

Evaluate and diagnose:
```
python tools/match.py runs/9x9_a/latest.pt anchors/AZNET9_epoch_300.pt --pairs 200 --visits 400
python tools/diagnose.py runs/9x9_a/latest.pt --val runs/9x9_a/val_buffer.npz
tensorboard --logdir runs
```

## Layout

| Path | What |
|---|---|
| `gozero/go/` | numba rules engine (Tromp-Taylor, positional superko), features, symmetries |
| `gozero/nn/` | ResNet with policy / value / score / ownership heads |
| `gozero/mcts/` | PUCT search batched across many games, evaluators (new net, legacy net, test players) |
| `gozero/selfplay/` | parallel self-play with playout cap randomization |
| `gozero/train/` | replay buffer (augmentation at sample time), trainer |
| `gozero/eval/` | arena with paired openings, Wilson intervals, Elo |
| `gozero/legacy/` | copies of the original rules / network for loading anchors and cross-checking |
| `gozero/loop.py` | the continuous training loop |
| `tools/` | `match.py` (evaluation), `diagnose.py` (component diagnostics) |
| `NOTES.md` | how to read the metrics and adjust hyperparameters |

## Differences from the original project

| Area | Original | Here |
|---|---|---|
| Policy target | visit counts *after* temperature (one-hot after move 18) | raw visit distribution |
| Root noise | never applied (bug) | Dirichlet at every self-play root with a full search |
| Net input | 8 boards of absolute history; root saw no history (bug) | relative stones, liberties, illegal points, last moves, "opponent passed" |
| Pass handling in search | ignored a pass made before the root | terminal detection uses the real pass count |
| Priors | floored at `MIN_PRIOR=0.01` | plain masked softmax |
| Ko | simple ko | positional superko |
| Speed | ~250 evals/s, one position per GPU call | ~7–17k evals/s (256 games batched, numba rules, CPU/GPU overlap) |
| Training scheme | gating arena, 16 games at 53% | continuous training, progress tracked against fixed anchors |
| Data reuse | ~40× per position (8 stored augmentations × passes) | ~4 samples per new sample, symmetry drawn at random |
| Extra targets | none | ownership and final score (KataGo) |
| Search budget | time-based | fixed visits, full/fast playout cap randomization |

## Interactive UI

```
python -m gozero.ui.server            # opens http://localhost:8765
python -m gozero.ui.server --model anchors/AZNET9_epoch_300.pt
```

A browser UI in the same style as the original (wooden board, translucent
candidate stones scaled by visits, red numbers, purple = best, green = close
alternatives, orange = last move), with more information:

| Mode | What it does |
|---|---|
| Analyze | live search on the current position (tree reused as you play); click to place stones |
| Play vs bot | choose your color and the bot's visits per move; hints off by default (press Analyze to turn them on) |
| Watch bots | any two checkpoints play each other, e.g. the new model vs `AZNET9_epoch_300` |
| Policy view | raw network move probabilities, no search |
| Sandbox | free play for testing rules; shows the Tromp-Taylor area score of the board |
| Puzzles | set up positions stone by stone, mark the answer, save to `problems/`, and test any model on one puzzle or all of them (move distribution at N visits, default 800) |

Side panel: black/white win-rate bar (search and raw net), score lead (score head),
top-moves table with win rate, visits, prior and principal variation (hover a row
to preview the line on the board, or hold Shift over a candidate), and a win-rate
graph over the game (click to jump). Overlays: search, policy, ownership (the
ownership head's territory estimate, plus markers on stones it thinks are dead),
move numbers, coordinates. Save / load SGF. Keys: ←/→, Home/End, P = pass,
Space = pause analysis. Old AZNet checkpoints work too (no score / ownership heads).
