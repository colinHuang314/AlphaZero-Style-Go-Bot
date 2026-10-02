# AlphaZero-Style Go Bot

A Go AI built from scratch following the AlphaGo Zero approach: a PyTorch policy/value network guiding Monte Carlo Tree Search, trained through self-play. I later adapted the same approach to chess: see [AlphaZero-Chess](https://github.com/colinHuang314/AlphaZero-Chess).

![Go analysis UI](AlphaZero-UI-Image2.png)

## Overview (`go/`)
- **Network:** ResNet (10 residual blocks, 128 channels) with policy, value, score and ownership heads; 16 input planes relative to the player to move (stones, group liberties, illegal points, last 4 moves)
- **Search:** PUCT MCTS batched across 256 games per GPU call, Dirichlet root noise, temperature schedule, playout cap randomization; numba rules engine (Tromp-Taylor scoring, positional superko)
- **Training:** continuous self-play (96,000+ games, ~110 hours on a laptop RTX 4060), with progress measured in matches against frozen earlier models
- **Results:** beats the original version of this bot 395–5 at equal search; 100–0 against GNU Go 3.8 (level 10); provisional CGOS 9×9 rating around 2,500–2,600
- **Interactive UI:** analyze positions with live MCTS, play the bot, watch two models play, inspect policy priors and ownership, and test life-and-death puzzles

## What I debugged
Temperature-schedule bugs, and eye-filling and self-atari loops.

## Development notes
I wrote the original Go engine myself from the AlphaGo Zero paper (see the early commit history). The improved version was built with AI-assisted development (Claude); I designed the architecture and training pipeline and debugged and evaluated the models.

## Try it
Weights: [Releases](https://github.com/colinHuang314/AlphaZero-Style-Go-Bot/releases/tag/v1.0)

```bash
pip install -r requirements.txt

# browser UI at http://localhost:8765
cd go
python -m gozero.ui.server --model path/to/go-zero-9x9-c439.pt
```
