# AlphaZero-Style Go Bot (9×9)

This project includes:
- An AlphaZero-style self-play training loop
- A model for 9x9 and 5x5 that I have trained for about 2 weeks of training time each
- An interactive UI that displays live MCTS search statistics and neural network priors of saved models

I started this project because of my interest in Go and my interest in game-playing AIs. I hope to develop more insight as to why the strength of the bots I trained plateaued. Although the bots didn't get to the strength I wanted, the training loop still worked, and the interactive UI looks so cool!

---

## Project Highlights
- AlphaZero-style **policy + value network**
- Full **self-play training loop** including evaluation against previous models
- Neural network–guided MCTS
- Interactive UI for visualizing:
  - MCTS visit counts
  - Policy priors
  - Other search statistics
- Extensive manual hyperparameter tuning
- Experimentation and failure analysis

---

## Current Performance
- Over ~3000 self-play games, the bot has learned incrementally, beating previous models
- Makes "intelligent" decisions under sufficient MCTS simulations
- After lots of training, performance seems to have plateaued on both **9×9 and 5x5** boards (2 different models)
- Performance is likely limited by:
  - Compute constraints
  - The complexity of the model or hyperparameter suboptimality
  - Training instability

---

<details>
  <summary><strong>Architecture Overview</strong></summary>

### Neural Network
- Input planes: **17** (1 for turn, 16 for the past 8 board positions binary encoded by player)
- Residual network:
  - **128 channels**
  - **14 residual blocks**
- Outputs:
  - Policy head (move probabilities)
  - Value head (win/loss prediction)

### Monte Carlo Tree Search (MCTS)
- Neural network–guided search
- CPUCT-based exploration
- Dirichlet noise during self-play
- Training and evaluation temperature schedules for variety vs performance

</details>

---

<details>
  <summary><strong>Training Framework</strong></summary>

- **AlphaZero-style self-play**
  - First. load the last saved model, champion model, and replay buffer
  - Then play self-play games, saving every game position, its augmentations (since the board is symmetric), the MCTS visit count distribution, and the winner of the game to the replay buffer
  - Optionally save model
  - Train model on replay buffer data every n self-play games
  - Evaluate the resulting model against the champion every m self-play games
    - If it wins, it replaces the champion and produces the next self-play games
    - If it loses, keep training

- Manual tuning of:
  - Number of iterations per game
  - Number of self-play games before evaluation (training with newer model)
  - Dirichlet constants
  - Max game length for early stages of training
  - Whether to use max length games in training
  - Number of arena games to play and the win rate needed to be promoted
  - Buffer size
  - Training epochs
  - *And more*

*Training was monitored for illegal moves, self-atari loops, max-length games, and value/policy imbalance.*

</details>

---

<details> <summary><strong>Observations & Training Issues</strong></summary>
  
### Issues I've had with training (past or present)
- Sometimes requires very high MCTS simulations to generate usable data

- Policy struggles with:
  - Captures
  - Eyes
  - Illegal moves (later mitigated)

- Training instability caused by:
  - Replay buffer mis-sizing
  - Temperature scheduling bugs
  - Max game length truncation corrupting data

- Bot often:
  - Plays overly safe
  - Fills its own eyes
  - Loops in self-atari under certain rare conditions

- Potential unresolved issues:
  - Gradient instability (possible ReLU explosion)
  - Value head overpowering policy
  - Replay buffer imbalance (too many of one player winning)

</details>

---

<details> <summary><strong>Lessons Learned</strong></summary>
  
- AlphaZero-style systems are sensitive to:
  - MCTS simulation count
  - Temperature schedules
  - Replay buffer composition

- Beating the current champion does not guarantee real improvement
- Poor self-play data compounds rapidly
- Limited compute severely restricts convergence even with a 9×9 board

**Despite these limitations, the system clearly demonstrates non-trivial learning behavior.**

</details>

---

<details> <summary><strong>Potential Future Work</strong></summary>
  
- Multithreaded or parallel self-play
- Faster rollout strategies
- Training without a fixed champion
- Improved illegal move handling
- Investigating gradient stability
- Bit-packed board representations
- Improved replay buffer sampling

</details>

---

# References
AlphaGo Zero
https://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf

AlphaZero
https://arxiv.org/pdf/1712.01815

Other resources:
- https://www.youtube.com/watch?v=0slFo1rV0EM
- https://www.youtube.com/watch?v=wuSQpLinRB4
- https://www.youtube.com/watch?v=UzYeqAJ2bA8

---
