# AlphaZero-Style-Go-Bot

<!--
Built a Go-playing AI inspired by AlphaZero, combining a PyTorch neural network with Monte Carlo Tree Search using guidance from the AlphaZero paper and related research.
Trained and evaluated the model by analyzing policy and value loss, self-play performance, tuning hyperparameters, and leveraging domain knowledge of Go to identify and address strategic weaknesses.
Developed an interactive UI to visualize live MCTS search statistics and neural network priors, and conducted modular testing to ensure system correctness and stability.



the bot i trained on 9x9 isn't very good, but it definitely has learned incrementally and makes somewhat intelligent decisions (most of the time)
im not sure if its my lack of computnig power, or something suboptimal in the code

it used the alphazero selfplay training framework

as i trained i tuned the parameters as i saw fit


The settings i used:
# Board
BOARD_SIZE = 9
N_PLANES = 17
AVG_MOVES_PER_GAME = 138

# Model
CHANNELS = 128
RESIDUAL_BLOCKS = 14

# Optimizer / LR
LR = 0.0008 # increase to escape minima[1e-3]
#ETA_MIN = 1e-5 
WEIGHT_DECAY = 2e-4 # L2 reg [1e-4]

# Overall
TRAIN_STEPS = 999  # just for saving selfplay games - can still run forever if never beats champ (epoch resets to champ)  runs however long you want(why not just manually stop it with key inturrupt)

# Self-play and MCTS --- [KOMI IN GOGAME.py]
SELFPLAY_GAMES_PER_ITER = 5#------     # how much time to spend
MCTS_SIMS = 2000 #---------                   # 200-400 ######################################### pi distribution affected, more sims = favors better moves more comparetively because of exploration formula
MIN_SEARCH_TIME = 12  # avg sims per second (100-250)+
# WILSON_LOWER_BOUND = 0.7  # with conf = 0.95
CPUCT = 4                      # [1.5+] exploration constant
MIN_PRIOR = 0.01      # [1/num_moves] this includes corner moves but gives moves the bot thinks are bad/illegal a chance (combat overfitting)
TEMPERATURE_SCHEDULE = [(1, 4), (1, 14), (1, 16), (1, 18), (0, 20), (0, 9999)]  # more temp with more sims is ok
DIRICHLET_ALPHA = 0.13   # flatten distribution (more noise)[0.13]
DIRICHLET_EPSILON = 0.2    # ratio of noise (more noise) [0.25]
MAX_GAME_LENGTH = 178 # (BOARD_SIZE * BOARD_SIZE * 2) + 40  # safety cap for moves in a game
MAX_ARENA_GAME_LENGTH = int(MAX_GAME_LENGTH * 1.5) # x2
USE_MAX_LENGTH_GAMES = False #######################--------------------------------################## 
RESIGNS = False #######################--------------------------------##################
WHITE_SIM_MULTIPLIER = 1

# Evaluation (Arena Games)
ARENA_GAMES = 20  # [16, 26, 50]   --need enough to cancel out randomness from temp--    # games the net plays against the champion for evaluation  
WINRATE_THRESHOLD = 0.53 # [0.53] winrate needed to replace champion
# IMMEDIATE_TERMINATION_THRESHOLD = 0.65 # loss rate to reset to champion
ARENA_INTERVAL = 100
ARENA_MCTS_SIMS = 2000  # [400] 
ARENA_MIN_SEARCH_TIME = 12 #[1.5]
TRAINING_ALLOWANCE = 9999 # not giving up --- num epochs until give up on curr training path and reset to chamption 
ARENA_TEMPERATURE_SCHEDULE = [(0.3 , 2), (0.15, 6), (0, 9999)] # just enough randomness to play different games
ARENA_DIRICHLET_ALPHA = 0.0
ARENA_DIRICHLET_EPSILON = 0.0


# Training from Buffer
MAX_BUFFER_SIZE = int(1.2 * 295 * (138 * 8)) #-- 430k --    # games * positions per game[320] # about games per itter[10] * avg moves per game[25-35] * augmentations[8] * num of past epochs to consider [10-20]
BATCH_SIZE = 64                  # 64 for accuracy, 256 for speed
                        # defined line 533 REPLAY_SAMPLES_PER_STEP = int(len(replay_buffer) / BATCH_SIZE * 5)     #------   ~ buffersize/batch size * num_repeats (avg)
TRAINING_EPOCHS = 4  # num times to train over the replay buffer(effectively, still using random sampling)
# MIN_SEED_GAMES = 32         # min games in the buffer to stabilize training early on
VALUE_LOSS_WEIGHT = 1.0  # [1.0] increase if model is bad for long-term play, decreace if value loss is too low (more budget goes to policy) (also for tactics)
MAX_NORM = 1.0 #for grad clipping 
TRAIN_INTERVAL = ARENA_INTERVAL ######/////////////

# Save/Logs
SAVE_INTERVAL = 1   # save every n training steps (make depending on how often you want to check on model/your memory)
TRAIN_LOG_INTERVAL = (MAX_BUFFER_SIZE / BATCH_SIZE * TRAINING_EPOCHS) // 11    # print info every n batches in a training step
SELFPLAY_GAME_LOG_INTERVAL = 1 #SELFPLAY_GAMES_PER_ITER//6




Conclusion
performance seems to plateau at both board sizes 5 and 9. this could be because of the complexitiy of the neural network or the resources I have to train
a peer told me about a potential gradient explosion with relu() but I haven't looked too much into it 


possible further steps: # check if you still need prior floor
# *** make self play faster with multithreading ***
# faster selfplay with other methods?
# try training without champ for a while?
# np.packbits?


past events when training: 
#boosted sims to stop maxlen games then dropped back
  #dropping back was a mistake, needed to go back about 10 days of training because the low sims messed it up
  #made sims even higher (1800+) to ensure good data
  #infinite loop with self atary on first line 2 stones, take and snapback
# changes MCTS to have binary win/loss instead of margin training (still one max game in selfplay) (now the bots never pass if they will lose(resign helps))
# arena games with more sims = less max mmove gaves(but max limit is double)
#bot seems to be playing it safe, but with 900+ sims, picks more optimal move
#policy not good with captures/eyes still, still makes illegal moves
#50 moves not enough cause if white takes black stones with throw ins, then black fills board fully except 2 eyes
#not improvong even though beating champion. maybe need more sims to improve further? bot is not better than epoch 630 or 280?
    # boosted sims x3 to 900 and now its going crazy with a 90% win then 100% eval win
# 730 generated more max len games than 710
# 730 playing self atari loop and playing in eyes
#wrong est for avg moves in game -> wrong buffersize, less games trained on, early games erased before training
# white wins more when temp is higher
#too many black wins in buffer?
#move confidence for faster moves?  BUT will mess up visit distrubution (dist could still change)
#add resignation (but currently MCTS value is almost always <95 since model fills in eyes and dies sometimes) (not need rn since with confidence, losing player will quickly pass )
#used boolean to store bit arrays
#epochs 680-711 were playing on the edge as white so reset to 670
#cpuct of 3 needed and min_prior of 0.04 (with 400 sims) to convince model to play "suicide" when it actually was a capture. 2.5 was not enough! (maybe enough if it was 1000 sims)
    # this worked on epoch 711 but not champ(670) the value network of the champ must be the reason
#only use unique augmentations? but positions will be weighted differently in training
#test black has 4 legal moves, 2 in eyes, 2 in opponents 2 groups eyes (test if live mcts will be correct after < 400 sims)
#removed illegal moves from policy before normalizing
# prior floor implemented (bot would assign huge capture that would normally be suicide with 0 prior)
#stoppd using max game length games in training
#temperature bug! was using moves made not move number so white was getting more varience
#max game len was too small! size^2 * 2 was not enough (soon stopped using max game length games in training)
#TRYING VOLUME AND HIGH TEMP 126s per epoch *FAIL
#trying precision 21 min per epoch 
# was decreasing temperature the issue?
# re work stop when confident
# more black wins in arena games
-->

## Sources I used for help
https://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf

https://arxiv.org/pdf/1712.01815

https://www.youtube.com/watch?v=0slFo1rV0EM

https://www.youtube.com/watch?v=wuSQpLinRB4

https://www.youtube.com/watch?v=UzYeqAJ2bA8
