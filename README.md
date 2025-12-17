# AlphaZero-Style-Go-Bot

Built a Go-playing AI inspired by AlphaZero, combining a PyTorch neural network with Monte Carlo Tree Search using guidance from the AlphaZero paper and related research.
Trained and evaluated the model by analyzing policy and value loss, self-play performance, tuning hyperparameters, and leveraging domain knowledge of Go to identify and address strategic weaknesses.
Developed an interactive UI to visualize live MCTS search statistics and neural network priors, and conducted modular testing to ensure system correctness and stability.

https://arxiv.org/abs/1712.01815

the bot i trained on 9x9 isn't very good, but it definitely has learned incrementally and makes somewhat intelligent decisions (most of the time)
im not sure if its my lack of computnig power, or something suboptimal in the code

it used the alphazero selfplay training framework
