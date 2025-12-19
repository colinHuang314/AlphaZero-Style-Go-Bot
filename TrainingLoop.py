import numpy as np
import random
import os
import warnings
import time
import copy
from datetime import datetime

# my files
from GoRules import Game  # has game rules
from MCTS_Go import MCTS
import BoardOperations_Go
#from GameEngineTesting import start_game_UI

# network
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

#aesthetics
# from tqdm import trange
from colorama import Fore
import winsound


# Board
BOARD_SIZE = 9
N_PLANES = 17
AVG_MOVES_PER_GAME = 160 # not used in training

# Model
CHANNELS = 128
RESIDUAL_BLOCKS = 14

# Optimizer / LR
LR = 0.001 # increase to escape minima[1e-3]
#ETA_MIN = 1e-5 
WEIGHT_DECAY = 1e-4 # L2 reg [1e-4]

# Overall
TRAIN_STEPS = 999  # just for saving selfplay games - can still run forever if never beats champ (epoch resets to champ)  runs however long you want(why not just manually stop it with key inturrupt)

'''--------------------------------------------'''
''' --------------- Settings ------------------'''
'''--------------------------------------------''' 

# Self-play and MCTS --- [KOMI IN GOGAME.py]
SELFPLAY_GAMES_PER_ITER = 6#------     # how much time to spend
MCTS_SIMS = 250 #---------                   # 200-400 | pi distribution affected, more sims = favors better moves more comparetively because of exploration formula
MIN_SEARCH_TIME = 1  # (not nessesary) avg sims per second (100-250)+
# WILSON_LOWER_BOUND = 0.7  # with conf = 0.95
CPUCT = 3                      # [1.5+] exploration constant
MIN_PRIOR = 0.01      # [1/num_moves] this includes corner moves but gives moves the bot thinks are bad/illegal a chance (combat overfitting)
TEMPERATURE_SCHEDULE = [(1, 4), (1, 14), (1, 16), (1, 18), (0, 20), (0, 9999)]  # more temp with more sims is ok
DIRICHLET_ALPHA = 0.13   # flatten distribution (more noise)[0.13]
DIRICHLET_EPSILON = 0.2    # ratio of noise (more noise) [0.25]
MAX_GAME_LENGTH = (BOARD_SIZE * BOARD_SIZE * 2) + 40  # safety cap for moves in a game [178]
MAX_ARENA_GAME_LENGTH = int(MAX_GAME_LENGTH * 2) # x2
USE_MAX_LENGTH_GAMES = True  # Put to false after bot completes 90% of games
RESIGNS = False # experimental
WHITE_SIM_MULTIPLIER = 1 # experimental [1 to disable]

# Evaluation (Arena Games)
ARENA_GAMES = 16  # [16, 26, 50]   --need enough to cancel out randomness from temp--    # games the net plays against the champion for evaluation  
WINRATE_THRESHOLD = 0.53 # [0.53] winrate needed to replace champion
# IMMEDIATE_TERMINATION_THRESHOLD = 0.65 # loss rate to reset to champion
ARENA_INTERVAL = 5
ARENA_MCTS_SIMS = 250  # [400] 
ARENA_MIN_SEARCH_TIME = 1 #[1.5]
TRAINING_ALLOWANCE = 9999 # experimental --- num epochs until give up on curr training path and reset to chamption 
ARENA_TEMPERATURE_SCHEDULE = [(0.3 , 2), (0.15, 6), (0, 9999)] # just enough randomness to play different games
ARENA_DIRICHLET_ALPHA = 0.0
ARENA_DIRICHLET_EPSILON = 0.0


# Training from Buffer
TRAIN_INTERVAL = ARENA_INTERVAL ######/////////////
MAX_BUFFER_SIZE = int(1.2 * SELFPLAY_GAMES_PER_ITER * TRAIN_INTERVAL * (160 * 8)) #-- 430k --    # games * positions per game[320] # about games per itter[10] * avg moves per game[25-35] * augmentations[8] * num of past epochs to consider [10-20]
BATCH_SIZE = 64                  # 64 for accuracy, 256 for speed (speed is bottle-necked by selfplay games anyway)
                        # defined line 533 REPLAY_SAMPLES_PER_STEP = int(len(replay_buffer) / BATCH_SIZE * 5)     #------   ~ buffersize/batch size * num_repeats (avg)
TRAINING_EPOCHS = 4  # num times to train over the replay buffer(effectively, still using random sampling)
# MIN_SEED_GAMES = 32         # min games in the buffer to stabilize training early on
VALUE_LOSS_WEIGHT = 1.0  # [1.0] increase if model is bad for long-term play, decreace if value loss is too low (more budget goes to policy) (also for tactics)
MAX_NORM = 1.0 #for grad clipping 

# Save/Logs
SAVE_INTERVAL = 1   # save every n training steps (make depending on how often you want to check on model/your memory)
TRAIN_LOG_INTERVAL = (MAX_BUFFER_SIZE / BATCH_SIZE * TRAINING_EPOCHS) // 11    # print info every n batches in a training step
SELFPLAY_GAME_LOG_INTERVAL = 1 #SELFPLAY_GAMES_PER_ITER//6


# Paths/device
CURR_EPOCH_PATH = fr"Projects\AZNET Cleaned Folder\paths\curr_epoch.txt"
CHAMPION_EPOCH_PATH = fr"Projects\AZNET Cleaned Folder\paths\champion_epoch.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

##############################################################################################################################################################################################
################################################################### --- NETWORK --- #########################################################################################################
##############################################################################################################################################################################################
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + x
        out = F.relu(out)
        return out

class AZNet9(nn.Module): # 621k parameters
    def __init__(self, in_planes=N_PLANES, channels=CHANNELS, blocks=RESIDUAL_BLOCKS,
                 board_size=BOARD_SIZE, num_groups=8):
        super().__init__()
        self.board_size = board_size

        # input block
        self.conv_in = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False)
        self.gn_in = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        # residual tower
        self.blocks = nn.Sequential(*[ResidualBlock(channels, num_groups) for _ in range(blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_gn = nn.GroupNorm(num_groups=1, num_channels=2)  # 2 channels, use GN with 1 group
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_gn = nn.GroupNorm(num_groups=1, num_channels=1)  # 1 channel, GN with 1 group
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.gn_in(out)
        out = F.relu(out)
        out = self.blocks(out)

        # policy head
        p = self.policy_conv(out)
        p = self.policy_gn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # value head
        v = self.value_conv(out)
        v = self.value_gn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v
##############################################################################################################################################################################################
##############################################################################################################################################################################################
##############################################################################################################################################################################################
'''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
##############################################################################################################################################################################################
#################################################################### --- Training --- #########################################################################################################
##############################################################################################################################################################################################
def get_temperature_for_move(move_number, temp_schedule):  # -15x + 22
    # move_number: 1-indexed move count
    for temp, upto in temp_schedule:
        if move_number <= upto:
            return temp
    return temp_schedule[-1][0]


def apply_temperature(pi, tau):   
    pi = np.array(pi, dtype=np.float64)  # ensure float
    if tau <= 1e-8:  # effectively zero, pick best move deterministically
        out = np.zeros_like(pi)
        out[np.argmax(pi)] = 1.0
        return out

    # Scale by 1/tau and re-normalize
    log_pi = np.log(np.clip(pi, 1e-12, 1.0))  # prevent log(0)
    scaled = np.exp(log_pi / tau)
    return scaled / np.sum(scaled)


def get_args(str):
    if str == "training":
        return {
            "BOARD_SIZE": BOARD_SIZE,
            "CPUCT": CPUCT,
            "MIN_PRIOR": MIN_PRIOR,
            "MCTS_SIMS": MCTS_SIMS,
            "TEMPERATURE_SCHEDULE": TEMPERATURE_SCHEDULE,
            'DIRICHLET_ALPHA': DIRICHLET_ALPHA,
            'DIRICHLET_EPSILON': DIRICHLET_EPSILON,
            'MAX_GAME_LENGTH': MAX_GAME_LENGTH,
            'DEVICE': DEVICE
        }
    elif str == "arena games":
        return {
            "BOARD_SIZE": BOARD_SIZE,
            "CPUCT": CPUCT,
            "MIN_PRIOR": MIN_PRIOR,
            "MCTS_SIMS": ARENA_MCTS_SIMS,
            "TEMPERATURE_SCHEDULE": ARENA_TEMPERATURE_SCHEDULE,
            'DIRICHLET_ALPHA': ARENA_DIRICHLET_ALPHA,
            'DIRICHLET_EPSILON': ARENA_DIRICHLET_EPSILON,
            'MAX_GAME_LENGTH': MAX_ARENA_GAME_LENGTH,
            'DEVICE': DEVICE
        }
    else:
        return {}
##############################################################################################################################################################################################
##############################################################################################################################################################################################
##############################################################################################################################################################################################
def test_arena_game(path1, path2, args, get_games=True, verbose=True):
    """
    Load two models from checkpoints and run an arena match.
    """
    # Load Champion  
    board_size = args['BOARD_SIZE']
    
    checkpoint1 = torch.load(path1, map_location="cpu", weights_only=False)
    champion = AZNet9(board_size=board_size)
    champion.load_state_dict(checkpoint1['model_state_dict'])
    champion.eval().to(args['DEVICE'])

    # Load Challenger  
    checkpoint2 = torch.load(path2, map_location="cpu", weights_only=False)
    challenger = AZNet9(board_size=board_size)
    challenger.load_state_dict(checkpoint2['model_state_dict'])
    challenger.eval().to(args['DEVICE'])

    # Run Arena  
    challenger_won, record, games = arena_match(champion, challenger, args, get_games=get_games, verbose=verbose)

    print("\n=== Arena Match Results ===")
    print(f"Champion (path {path1}) vs Challenger (path {path2})")
    print(f"{Fore.YELLOW}Score: Champion: {record[0]}, Challenger: {record[1]}{Fore.RESET}")
    print(f"CHALLENGER {'PROMOTED' if challenger_won else 'NOT PROMOTED'}")
    return challenger_won, games
##############################################################################################################################################################################################
##############################################################################################################################################################################################
##############################################################################################################################################################################################

def self_play_game(model, args, verbose=False):
    """
    Run one self-play game using MCTS + model.
    Returns a list of (state, pi, z) training samples.
    """

    # Initialization  
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

    to_play = 1   # 1 = Black, -1 = White
    data_history = []
    board_history = [board.copy()]
    action_taken = None
    prev_board = None

    hope1 = 3
    hope2 = 3
    
    move_number = 1
    consecutive_passes = 0
    temp = 1
    while True:
         # Run MCTS from current position  
        mcts = MCTS(model, board, to_play, args, prev_board, debug=False)

        pi, move_value = mcts.search(MIN_SEARCH_TIME * (WHITE_SIM_MULTIPLIER if to_play == -1 else 1))
        if RESIGNS:
            if to_play == 1:
                if move_value <= -0.95:
                    hope1 -= 1
                elif hope1 < 3: 
                    hope1 += 1
            else:
                if move_value <= -0.95:
                    hope2 -= 1
                elif hope2 < 3: 
                    hope2 += 1
         # Apply temperature schedule  
        temp = get_temperature_for_move(move_number, args['TEMPERATURE_SCHEDULE'])
        pi = apply_temperature(pi, temp) # argmax for t = 0

        # Select move  
        action_idx = np.random.choice(len(pi), p=pi)
        action = Game.index_to_action(action_idx, args['BOARD_SIZE'])
        action_taken = action

        # Record training data (before applying move)  

        state_planes = BoardOperations_Go.make_input_planes(board_history, to_play, args['BOARD_SIZE'])  # shape: [N_PLANES, board_size, board_size]
        data_history.append((state_planes, pi, to_play))  # we’ll attach final outcome later
        # Add augmented versions  
        for i in range(1, 8):  # skip identity (0)
            # Apply symmetry to each board in history
            sym_history = [BoardOperations_Go.transform_board(b, i) for b in board_history[-8:]]

            # Rebuild input planes from symmetric history
            sym_planes = BoardOperations_Go.make_input_planes(sym_history, to_play, args['BOARD_SIZE'])

            # Apply same symmetry to pi
            sym_pi = BoardOperations_Go.invert_sym_pi(pi, i, args['BOARD_SIZE'])

            # Store augmented training data
            data_history.append((sym_planes, sym_pi, to_play))


        # Apply move  
        prev_board = board.copy()

        if action is None:  # "pass"
            consecutive_passes += 1
        else:
            board = Game.apply_move(board, action, to_play, args['BOARD_SIZE'])
            consecutive_passes = 0

        board_history.append(board.copy())
        if len(board_history) > 8:
            board_history.pop(0)  # keep only the last 8

        to_play *= -1
        move_number += 1

        # Check end conditions  
        if consecutive_passes >= 2:
            if move_number >= args['MAX_GAME_LENGTH']:
                print(Fore.RED + "mx-" + Fore.RESET, end="")
                #print(f"\n{board}")    
            break
        if move_number >= args['MAX_GAME_LENGTH']:
            print(Fore.RED + "mx-" + Fore.RESET, end="")
            #print(f"\n{board}")
            break
        if RESIGNS and (hope1 <= 0 or hope2 <= 0):
            print(Fore.RED + "res-" + Fore.RESET, end = "")
            break

    # Score final outcome  
    if RESIGNS and (hope1 <= 0 or hope2 <= 0): score = to_play
    else: score, _ = Game.final_score_tromp_taylor(board, args['BOARD_SIZE'])  # +1 black win, -1 white win, 0 draw
    
    if verbose: 
        if score > 0: print("s=" + Fore.BLACK + str(score) + Fore.RESET, end=" ")
        else: print("s=\033[1m" + str(score) + "\033[0m", end=" ")

    if score > 0: winner = 1
    else: winner = -1

    # Build training samples  
    data = []
    for state_planes, pi, player in data_history:
        z = winner * player  # flip perspective
        data.append((state_planes, pi, z))

    return data, winner, move_number

def arena_match(champion, challenger, args, get_games=False, verbose=False):
    champion_color = 1
    champion_wins = 0
    challenger_wins = 0
    games = []

    for _ in range(ARENA_GAMES):
        # Initialization  
        board = np.zeros((args['BOARD_SIZE'], args['BOARD_SIZE']), dtype=np.int8)
        to_play = 1   # 1 = Black, -1 = White
        board_history = [board.copy()]
        prev_board = None
        move_number = 1
        consecutive_passes = 0

        hope = 3
        temp = 1
        while True:
            # Run MCTS from current position  
            model = champion if (champion_color * to_play == 1) else challenger
            mcts = MCTS(model, board, to_play, args, prev_board, debug=False)
            pi, move_value = mcts.search(ARENA_MIN_SEARCH_TIME)
            
            # Apply temperature schedule  
            temp = get_temperature_for_move(move_number, args['TEMPERATURE_SCHEDULE'])
            pi = apply_temperature(pi, temp)

            # Select move  
            action_idx = np.random.choice(len(pi), p=pi)
            action = Game.index_to_action(action_idx, args['BOARD_SIZE'])
            
            # Apply move  
            prev_board = board.copy()
            if action is None:  # "pass"
                consecutive_passes += 1
            else:
                board = Game.apply_move(board, action, to_play, args['BOARD_SIZE'])
                consecutive_passes = 0

            if get_games: board_history.append(board.copy())

            to_play *= -1
            move_number += 1

            # Check end conditions  
            if consecutive_passes >= 2:
                if verbose: print("Arena Game ended by passing, move: " + str(move_number), end=", ")
                break
            elif move_number >= args['MAX_GAME_LENGTH']:
                if verbose: print("Arena Game ended by move limit, ", end="")
                break
         
        # Score final outcome  
        if RESIGNS and hope <= 0:
            score = to_play
        else: score, _ = Game.final_score_tromp_taylor(board, args['BOARD_SIZE'])  # +1 black win, -1 white win, 0 draw
        if verbose: print("score: " + (Fore.BLACK if score > 0 else "") + str(score) + Fore.RESET, end=", ")
        if score > 0: winner = 1
        else: winner = -1

        if (winner * champion_color) == 1:
            champion_wins += 1
            print(f"Champion wins. Record: ({champion_wins}, {challenger_wins})")
        else:
            challenger_wins += 1
            print(f"Challenger wins. Record: ({champion_wins}, {challenger_wins})")

        
        # Optional early stop  
        if challenger_wins > ARENA_GAMES * WINRATE_THRESHOLD:
            break
        if champion_wins > ARENA_GAMES * (1 - WINRATE_THRESHOLD):
            break

        champion_color *= -1 #switch colors

        if get_games: games.append(board_history)

    print(" --- Results ---")
    if challenger_wins + champion_wins == 0:
        win_rate = 0
    else: win_rate = challenger_wins / (challenger_wins + champion_wins)
    print(f"Champion wins: {champion_wins}, Challenger wins: {challenger_wins}, total games: {champion_wins+challenger_wins}, WINRATE: {win_rate:.2f}")

    if win_rate >= WINRATE_THRESHOLD:
        print("Challenger overtakes")
        return True, [champion_wins, challenger_wins], games
    else: 
        print("Champion holds")
        return False, [champion_wins, challenger_wins], games
    
def train_loop():
    time_tracker = [time.time()]
    champion = AZNet9().to(DEVICE) # for self play and eval (arena match)
    model = AZNet9().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler() # ignore warning

    # scheduler = CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=ETA_MIN)

    args = get_args("training")
    arena_args = get_args("arena games")
    champion_epoch = 0 # champion epoch - for loading and training allowance
    last_loaded_epoch = 0 # just for loading
    epoch = 1
    replay_buffer = []  # list of (board_planes, pi, z)

    with open(CHAMPION_EPOCH_PATH, "r") as f: #get champion file
        champion_epoch = int(f.read().strip())
    champion_path = fr"Projects\AZNET Cleaned Folder\paths\AZNET{BOARD_SIZE}_checkpoint_epoch_{champion_epoch}.pt" # AZNET5_checkpoint_epoch_XX 

    with open(CURR_EPOCH_PATH, "r") as f: # get curr file
        last_loaded_epoch = int(f.read().strip())
    last_loaded_path = fr"Projects\AZNET Cleaned Folder\paths\AZNET{BOARD_SIZE}_checkpoint_epoch_{last_loaded_epoch}.pt" # AZNET5_checkpoint_epoch_XX 

    if os.path.exists(champion_path): # load champion
        print(f"Loading champion: {champion_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(champion_path, map_location=DEVICE, weights_only=False)
        champion.load_state_dict(checkpoint['model_state_dict'])

    else: print("No champion checkpoint found, starting fresh.")

    if os.path.exists(last_loaded_path): # load curr chackpoint
        print(f"Loading curr: {last_loaded_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            checkpoint = torch.load(last_loaded_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        replay_buffer = checkpoint['replay_buffer']
    else:
        model = copy.deepcopy(champion).to(DEVICE) ####################
        print("curr model not found, starting from champion")


    now = datetime.now()
    formatted_time = now.strftime("%m/%d/%y %I:%M %p").lower();formatted_time = formatted_time.replace('/0', '/').replace(' 0', ' ')
    print(formatted_time)
    #######################################
    ############# - LOOP -#################
    epoch = last_loaded_epoch + 1 ################################

    while epoch <= last_loaded_epoch + TRAIN_STEPS:
        print(f"{Fore.GREEN}Epoch {epoch}/{last_loaded_epoch + TRAIN_STEPS} | buffer len: {len(replay_buffer)}{Fore.RESET}")

        # Self-play games   BY CHAMPION
        black_wins = 0
        white_wins = 0
        max_length_games = 0
        for g in range(SELFPLAY_GAMES_PER_ITER):
            if g < SELFPLAY_GAMES_PER_ITER:
                selfplay_start = time.time()
            data, winner, moves = self_play_game(champion, args, verbose = (g+1) % SELFPLAY_GAME_LOG_INTERVAL == 0)  # returns list of (state_planes, pi, z)
            

            if moves < MAX_GAME_LENGTH:
                replay_buffer.extend(data)
                if winner == 1: black_wins += 1
                else: white_wins += 1
            else:
                max_length_games += 1
                if USE_MAX_LENGTH_GAMES:
                    replay_buffer.extend(data)
                    if winner == 1: black_wins += 1
                    else: white_wins += 1


            if g < SELFPLAY_GAMES_PER_ITER:
                print(f"game {g} took: {round(time.time() - selfplay_start)} for {moves} moves")
        print(f"black wins: {black_wins}, white wins: {white_wins}, max length games: {max_length_games}")

        # Trim buffer if necessary
        if len(replay_buffer) > MAX_BUFFER_SIZE:
            replay_buffer = replay_buffer[-MAX_BUFFER_SIZE:]

        winsound.Beep(400, 200)  #  Hz for time 
        winsound.Beep(400, 200)

        # Save checkpoint  
        if epoch % SAVE_INTERVAL == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'replay_buffer': replay_buffer,   # save buffer too
                'epoch': epoch,
            }, fr"Projects\AZNET Cleaned Folder\paths\AZNET{BOARD_SIZE}_checkpoint_epoch_{epoch}.pt")

            print(f"\t\t\t\t\t\t\t\t\t{Fore.YELLOW}\033[1mCheckpoint saved at epoch {epoch}\033[0m{Fore.RESET}")
            with open(CURR_EPOCH_PATH, "w") as f:
                f.write(str(epoch))


        if epoch % TRAIN_INTERVAL == 0:
            print("Starting training")
            training_start = time.time()
            # Sample training batches  
            REPLAY_SAMPLES_PER_STEP = int(len(replay_buffer) / BATCH_SIZE * TRAINING_EPOCHS) ####################################################################################################################################
            for step in range(REPLAY_SAMPLES_PER_STEP):
                batch_samples = random.sample(replay_buffer, min(BATCH_SIZE, len(replay_buffer)))

                # Unpack batch into tensors
                states = torch.tensor(np.array([s.astype(np.float32) for s, _, _ in batch_samples]), dtype=torch.float32, device=DEVICE)
                target_pis = torch.tensor(np.array([p for _, p, _ in batch_samples]), dtype=torch.float32, device=DEVICE)
                target_vs = torch.tensor(np.array([z for _, _, z in batch_samples]), dtype=torch.float32, device=DEVICE)

                with autocast():   # mixed precision forward

                    # Forward pass  
                    pred_pis, pred_vs = model(states)  # pred_pis: (B, action_size), pred_vs: (B,)

                    # Compute losses  
                    # Policy: cross-entropy between distributions
                    log_probs = F.log_softmax(pred_pis, dim=1)
                    policy_loss = -(target_pis * log_probs).sum(dim=1).mean()

                    # Value: MSE
                    value_loss = F.mse_loss(pred_vs, target_vs)

                    loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

                # Backpropagation  
                opt.zero_grad()
                scaler.scale(loss).backward()   # scaled backward
                # gradient clipping works after unscale
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), MAX_NORM)
                scaler.step(opt)    # optimizer step (maybe skipped if inf/NaN detected)
                scaler.update()     # update scale for next iteration

                # scheduler.step()

                if step % TRAIN_LOG_INTERVAL == 0 or step == REPLAY_SAMPLES_PER_STEP - 1:
                    print(f"T: {loss.item():.2f}, "
                        f"{Fore.LIGHTCYAN_EX}P: {policy_loss.item():.2f}, {Fore.RESET}"
                        f"{Fore.MAGENTA}V: {value_loss.item():.2f} (*{VALUE_LOSS_WEIGHT}){Fore.RESET}", end = "| ")
            print()
            print(f"Training took: {round(time.time() - training_start)}")


        if epoch % ARENA_INTERVAL == 0:
            print("Starting arena games")
            arena_start = time.time()
            # Arena games (after save model)  
            challenger_won, record, _ = arena_match(champion, model, arena_args, verbose=True)
            print(f"\n{Fore.YELLOW}=== Arena Match Results ==={Fore.RESET}")
            print(f"Champion (path {champion_epoch}) vs Challenger (path {epoch})")
            print(f"Score: Champion: {record[0]}, Challenger: {record[1]}")
            print(f"\033[1mCHALLENGER {f"{Fore.GREEN}PROMOTED\033[0m" if challenger_won else f"{Fore.RED}NOT PROMOTED"}{Fore.RESET}")
            
            if challenger_won:
                with open(CHAMPION_EPOCH_PATH, "w") as f:
                    f.write(str(epoch))
                champion = copy.deepcopy(model).to(DEVICE) ####################
                champion_epoch = epoch

            # reset model if not improved in too long
            over_training_allowance = epoch - champion_epoch >= TRAINING_ALLOWANCE
            # lost_badly = (record[0] / sum(record)) > IMMEDIATE_TERMINATION_THRESHOLD
            if over_training_allowance:
                # if lost_badly: print("lost badly")
                print(f"{Fore.RED}\033[1mChallenger reset at epoch {epoch}, re-syncing with champion.\033[0m{Fore.RESET}")
                epoch = champion_epoch
                model = copy.deepcopy(champion).to(DEVICE) ####################
                opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                scaler = GradScaler() # ignore warning
            print(f"Arena game took: {round(time.time() - arena_start)}")


        time_tracker.append(time.time()) # get time

        # Current time
        now = datetime.now()
        # Format: M/D/YY h:mm am/pm
        formatted_time = now.strftime("%m/%d/%y %I:%M %p").lower()
        formatted_time = formatted_time.replace('/0', '/').replace(' 0', ' ')

        print(f"\t{Fore.BLUE}time for this epoch: {round(time_tracker[-1] - time_tracker[-2])} | average time per epoch: {round((time_tracker[-1] - time_tracker[0])/(len(time_tracker)-1))}{Fore.RESET} - {formatted_time}")

        
        epoch += 1

    print(f"{Fore.BLUE}total time taken: {(time_tracker[-1] - time_tracker[0]):.2f} for {TRAIN_STEPS} epochs{Fore.RESET}")


# MAIN

print(f"Estimated time for one cycle:  {(MIN_SEARCH_TIME * (0.5 * (WHITE_SIM_MULTIPLIER + 1)) * AVG_MOVES_PER_GAME * SELFPLAY_GAMES_PER_ITER * ARENA_INTERVAL + ARENA_MIN_SEARCH_TIME * AVG_MOVES_PER_GAME * ARENA_GAMES) / 3600:.2f}")
with open(CURR_EPOCH_PATH, "r") as f:
    last_loaded_epoch = int(f.read().strip())
epochs_left = ARENA_INTERVAL - (last_loaded_epoch % ARENA_INTERVAL)
runtime = (MIN_SEARCH_TIME * (0.5 * (WHITE_SIM_MULTIPLIER + 1)) * AVG_MOVES_PER_GAME * SELFPLAY_GAMES_PER_ITER * epochs_left) / 3600
eval_time = (ARENA_MIN_SEARCH_TIME * AVG_MOVES_PER_GAME * ARENA_GAMES) / 3600
print(f"estimated runtime left: {(runtime):.2f} + {(eval_time):.2f} = {(runtime + eval_time):.2f} HOURS")
print("For a better estimate see how long it takes for the first game.")

train_loop()


####################################################################################################################################################
r'''                                                        EXAMPLE OUTPUT:

Using device: cuda
Estimated time for one cycle:  1.88
estimated runtime left: 1.17 + 0.21 = 1.38 HOURS
For a better estimate see how long it takes for the first game.

No champion checkpoint found, starting fresh.
Loading curr: Projects\AZNET Cleaned Folder\paths\AZNET9_checkpoint_epoch_3.pt
12/18/25 3:32 pm
Epoch 4/1002 | buffer len: 12664
s=-0.5 game 0 took: 189 for 120 moves
s=-7.5 game 1 took: 177 for 120 moves
s=-9.5 game 2 took: 175 for 115 moves
s=21.5 game 3 took: 253 for 166 moves
black wins: 1, white wins: 3, max length games: 0
                                                                        Checkpoint saved at epoch 4
        time for this epoch: 796 | average time per epoch: 796 - 12/18/25 3:46 pm
Epoch 5/1002 | buffer len: 16800
s=12.5 game 0 took: 200 for 126 moves
s=-2.5 game 1 took: 178 for 120 moves
s=2.5 game 2 took: 234 for 158 moves
mx-s=-5.5 game 3 took: 319 for 202 moves
black wins: 2, white wins: 2, max length games: 1
                                                                        Checkpoint saved at epoch 5
        time for this epoch: 932 | average time per epoch: 864 - 12/18/25 4:01 pm
Epoch 6/1002 | buffer len: 21616
s=-6.5 game 0 took: 189 for 117 moves
s=7.5 game 1 took: 187 for 124 moves
s=2.5 game 2 took: 159 for 102 moves
mx-s=5.5 game 3 took: 312 for 202 moves
black wins: 3, white wins: 1, max length games: 1
                                                                        Checkpoint saved at epoch 6
        time for this epoch: 847 | average time per epoch: 859 - 12/18/25 4:15 pm
Epoch 7/1002 | buffer len: 25944
s=-38.5 game 0 took: 271 for 174 moves
s=-5.5 game 1 took: 198 for 130 moves
s=-5.5 game 2 took: 239 for 145 moves
s=-34.5 game 3 took: 215 for 141 moves
black wins: 0, white wins: 4, max length games: 0
                                                                        Checkpoint saved at epoch 7
        time for this epoch: 925 | average time per epoch: 875 - 12/18/25 4:31 pm
Epoch 8/1002 | buffer len: 30632
s=-2.5 game 0 took: 185 for 120 moves
s=-4.5 game 1 took: 199 for 126 moves
s=-36.5 game 2 took: 256 for 151 moves
s=-9.5 game 3 took: 181 for 124 moves
black wins: 0, white wins: 4, max length games: 0
                                                                        Checkpoint saved at epoch 8
        time for this epoch: 822 | average time per epoch: 865 - 12/18/25 4:44 pm
Epoch 9/1002 | buffer len: 34768
s=6.5 game 0 took: 183 for 123 moves
s=0.5 game 1 took: 226 for 143 moves
s=11.5 game 2 took: 163 for 100 moves
s=20.5 game 3 took: 228 for 145 moves
black wins: 4, white wins: 0, max length games: 0
                                                                        Checkpoint saved at epoch 9
        time for this epoch: 801 | average time per epoch: 854 - 12/18/25 4:58 pm
Epoch 10/1002 | buffer len: 38824
s=4.5 game 0 took: 182 for 125 moves
s=5.5 game 1 took: 172 for 109 moves
s=7.5 game 2 took: 228 for 153 moves
s=12.5 game 3 took: 289 for 183 moves
black wins: 4, white wins: 0, max length games: 0
                                                                        Checkpoint saved at epoch 10
Starting training
T: 5.41, P: 4.41, V: 1.00 (*1.0)| T: 5.01, P: 4.12, V: 0.88 (*1.0)| T: 4.19, P: 3.63, V: 0.56 (*1.0)| T: 4.13, P: 3.90, V: 0.24 (*1.0)| T: 4.11, P: 3.91, V: 0.20 (*1.0)| T: 4.09, P: 3.87, V: 0.22 (*1.0)| T: 3.90, P: 3.74, V: 0.16 (*1.0)| T: 3.86, P: 3.67, V: 0.18 (*1.0)|
Training took: 65
Starting arena games
Arena Game ended by passing, move: 102, score: 8.5, Champion wins. Record: (1, 0)
Arena Game ended by passing, move: 135, score: 35.5, Challenger wins. Record: (1, 1)
Arena Game ended by move limit, score: 0.5, Champion wins. Record: (2, 1)
Arena Game ended by passing, move: 134, score: 36.5, Challenger wins. Record: (2, 2)
Arena Game ended by passing, move: 69, score: 7.5, Champion wins. Record: (3, 2)
Arena Game ended by passing, move: 122, score: 32.5, Challenger wins. Record: (3, 3)
Arena Game ended by passing, move: 81, score: 0.5, Champion wins. Record: (4, 3)
Arena Game ended by passing, move: 140, score: 51.5, Challenger wins. Record: (4, 4)
Arena Game ended by passing, move: 142, score: -2.5, Challenger wins. Record: (4, 5)
Arena Game ended by passing, move: 107, score: 1.5, Challenger wins. Record: (4, 6)
 --- Results ---
Champion wins: 4, Challenger wins: 6, total games: 10, WINRATE: 0.60
Challenger overtakes

=== Arena Match Results ===
Champion (path 0) vs Challenger (path 10)
Score: Champion: 4, Challenger: 6
CHALLENGER PROMOTED
Arena game took: 2474
        time for this epoch: 3412 | average time per epoch: 1219 - 12/18/25 5:55 pm

'''

