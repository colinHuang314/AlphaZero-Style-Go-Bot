import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button

import torch
import torch.nn as nn
import torch.nn.functional as F

from GoRules import Game
import BoardOperations_Go
from MCTS_Go import MCTS
import MCTS_Go_Live

''' 
Modes:
    "sandbox"
    "view policy"
    "live mcts"   <----
    "play game"
'''
mode = "view policy" 

SEARCH_TIME = 3 # for "play game"
player_goes_first = True # for "play game"

BOARD_SIZE = 9 # if you change, make sure the path has a model that matches the board size
path = r"ExampleFolder\AZNET9_epoch_300.pt" # change to your path (needs to match board size)
# path = r"Projects\AZNET Cleaned Folder\AZNET5_epoch_970.pt" # change to your path (needs to match board size)

# not recommended to change the rest
N_PLANES = 17 

if BOARD_SIZE == 5:
    CHANNELS = 64
    RESIDUAL_BLOCKS = 8
elif BOARD_SIZE == 9:
    CHANNELS = 128
    RESIDUAL_BLOCKS = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")

ARENA_TEMPERATURE_SCHEDULE = [(0.3, 2), (0.15, 4), (0, 9999)]

args = {
    "BOARD_SIZE": BOARD_SIZE,
    "CPUCT": 3,
    "MIN_PRIOR": 0.01,
    "MCTS_SIMS": 100, # not used
    "MIN_SEARCH_TIME": SEARCH_TIME,
    "TEMPERATURE_SCHEDULE": ARENA_TEMPERATURE_SCHEDULE,
    'DIRICHLET_ALPHA': 0,  
    'DIRICHLET_EPSILON': 0,
    'MAX_GAME_LENGTH': BOARD_SIZE*BOARD_SIZE*20, # not used
    'DEVICE': DEVICE
}
#############################################################################################################################################################################################
################################################################### --- NETWORK --- #########################################################################################################
#############################################################################################################################################################################################
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

class AZNet9(nn.Module):
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
        self.policy_gn = nn.GroupNorm(num_groups=1, num_channels=2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_gn = nn.GroupNorm(num_groups=1, num_channels=1)
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
def get_temperature_for_move(move_number, temp_schedule):
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
##############################################################################################################################################################################################
##############################################################################################################################################################################################
##############################################################################################################################################################################################
def get_policy_value(model, board_history, to_play, size, device):
        model.eval()

        input = BoardOperations_Go.make_input_planes(
            board_history, to_play, size
        )
        input = input.astype(np.float32)

        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)

        if input.ndim == 3:
            input = input.unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = model(input)

        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        exp = np.exp(policy_logits - np.max(policy_logits))
        probs = exp / np.sum(exp)

        policy = []
        for idx, prob in enumerate(probs):
            if idx == size * size:  # last index reserved for pass move
                action = None
            else:
                r, c = divmod(idx, size)
                action = (r, c)
            policy.append((action, float(prob)))  # ensure Python float

        value = float(value.item())  # scalar float

        return policy, value

def display_board(ax, board, history, history_index, prev_move=None, policy=[], 
                  value_est=None, MCTS_visits=[], MCTS_probs=[], MCTS_value=0.0, 
                  ghost=None, to_play=None, game_over=False, score=None, winner=None,
                    territory=[], play_bot=False, player_move=True):
    size = board.shape[0]
    ax.clear()
    ax.set_aspect('equal')
    ax.add_artist(plt.Rectangle((-0.5, -0.5), 9, 9, color="#C18D56", zorder=1))#type:ignore
   
    # Draw grid
    for x in range(size):
        ax.plot([0, size-1], [x, x], color="black", zorder=1)
        ax.plot([x, x], [0, size-1], color="black", zorder=1)

    # Stones
    for i in range(size):
        for j in range(size):
            if board[i, j] == 1:  # black
                ax.add_artist(plt.Circle((j, i), 0.4, color="black", zorder=2)) #type:ignore
            elif board[i, j] == -1:  # white
                ax.add_artist(plt.Circle((j, i), 0.4, color="white", ec="black", zorder=2))#type:ignore

    # prev_move
    if prev_move:
        ax.add_artist(plt.Rectangle((prev_move[1] - 0.15, prev_move[0] - 0.15), 0.3, 0.3, color="orange", alpha=1.0, zorder=2))#type:ignore
    elif len(history) > 1: #pass
        ax.add_artist(plt.Rectangle((size/2 - 2, size-0.7), 3, 0.3, color="orange", alpha=1.0, zorder=3))#type:ignore

    # policy
    if policy:
        max_prob = max(policy, key=lambda x: x[1])[1]

        # get all actions with visits >= 75% of max
        good_actions = [action for action, prob in policy if prob >= 0.75 * max_prob]

        # best action is still the one with max visits
        best_action = max(policy, key=lambda x: x[1])[0]


        if len(history)>1:prev_board = history[-2]
        else:prev_board = None
        if not Game.is_legal(board, best_action, to_play, prev_board, size): # illegal move
            ax.add_artist(plt.Circle((best_action[1], best_action[0]), 0.2, color="red", zorder=3))#type:ignore

        for action, prob in policy:
            if Game.is_legal(board, action, to_play, prev_board, size):
                if action == best_action:
                    weight = 'extra bold'
                    font_size = 12 
                    ec = "purple"
                elif action in good_actions:
                    weight = 'bold'
                    font_size = 10 
                    ec = "green"
                else: 
                    font_size = 8
                    weight = 'normal'
                    ec = None if to_play else "black"

                if action != None:
                    if prob > 0.001:
                        alpha = (prob / max_prob) ** 0.9 * 0.8
                        if to_play == 1: ax.add_artist(plt.Circle((action[1], action[0]), 0.4, color="black", ec=ec, alpha=alpha, zorder=3))#type:ignore
                        else: ax.add_artist(plt.Circle((action[1], action[0]), 0.4, color="white", ec=ec, alpha=alpha, zorder=3))#type:ignore

                        prob_text = f"{prob:.3f}"
                        ax.text(action[1], action[0], prob_text,fontsize=font_size,ha="center", va="bottom",color="red",fontweight = weight)
                else: # pass
                    prob_text = f"{prob:.3f}"
                    ax.text(size/2, -1,  prob_text,fontsize=(12 + 4 * (weight == 'extra bold')),ha="center", va="bottom",color="red",fontweight = weight)

    if to_play == 1: player_str = "black"  
    else: player_str = "white"      

    if value_est:
        value_est_text = f"Policy Value: {value_est:.3f} | ({round((value_est + 1)/2 * 100, 2)}%) (for {player_str})"
        ax.text(
            size/2-1, size - 0.15,  
            value_est_text,
            fontsize=10, 
            ha="center", va="bottom",
            color="red",
            fontweight="bold"
        )

    if play_bot:
        ax.text(size/2-0.3, size+0.1, "Game vs Bot", fontsize=16,ha="center", va="bottom",color="blue",fontweight='bold')
        if player_move: info_text = f"wait for bot move ({SEARCH_TIME}s)"
        else: info_text = "click a cross-section once and wait for bot move"
        ax.text(size/2 - 0.5, size/2 + 4, info_text, fontsize=10,ha="center", va="bottom",color="black",fontweight='bold')

    # visits/ MCTS probs (normalized visits)
    if len(MCTS_visits) > 0 and len(MCTS_probs) > 0:
        total_sims_text = f"{sum(MCTS_visits)}" # total sims
        ax.text(size/2-0.3, size+0.1, total_sims_text,fontsize=16,ha="center", va="bottom",color="blue",fontweight='bold')

        MCTS_value_text = f"MCTS Value: {MCTS_value:.3f} | ({round((MCTS_value + 1)/2 * 100, 2)}%) (for {player_str})"
        ax.text(size/2-1, size/2 + 4, MCTS_value_text,fontsize=10,ha="center", va="bottom",color="black",fontweight='bold')


        best_action = Game.index_to_action(MCTS_visits.index(max(MCTS_visits)), size)
        max_prob = max(MCTS_probs)
        good_actions = [Game.index_to_action(MCTS_probs.index(prob), size) for prob in MCTS_probs if prob >= 0.7 * max_prob]


        for i in range(size * size+1):
            action = Game.index_to_action(i, size)
            if action == best_action:
                weight = 'extra bold'; font_size = 12; ec = 'purple'; stone_size = 0.44
            elif action in good_actions:
                weight = 'extra bold'; font_size = 9; ec = 'green'; stone_size = 0.41
            else: 
                font_size = 7; weight = 'normal'; ec = None if to_play else "black"; stone_size = 0.38

            if action != None:
                if MCTS_probs[i] > (1.0/2/size/size):
                    alpha = (MCTS_probs[i] / max_prob) ** 0.9 * 0.8
                    if to_play == 1: ax.add_artist(plt.Circle((action[1], action[0]), stone_size, color="black", ec=ec, alpha=alpha, zorder=3))#type:ignore
                    else: ax.add_artist(plt.Circle((action[1], action[0]), stone_size, color="white", ec=ec, alpha=alpha, zorder=3))#type:ignore

                    prob_text = f"{MCTS_probs[i]:.3f}"
                    ax.text(action[1], action[0], prob_text,fontsize=font_size,ha="center", va="bottom",color="red",fontweight=weight)
                    visit_text = f"{int(MCTS_visits[i])}"
                    ax.text(action[1], action[0] - 0.25, visit_text,fontsize=font_size,ha="center", va="bottom",color="red",fontweight=weight)
            else:
                prob_text = f"{MCTS_probs[i]:.3f},"
                ax.text(size/2 - 1, -1,  prob_text,fontsize=(8 + 4 * (weight == 'extra bold')),ha="center", va="bottom",color="red",fontweight = weight)
                visit_text = f"{MCTS_visits[i]:.3f}"
                ax.text(size/2+0.3, -1,  visit_text,fontsize=(8 + 4 * (weight == 'extra bold')),ha="center", va="bottom",color="red",fontweight = weight)

    # Ghost stone (only if game not over)
    if ghost is not None and not game_over:
        gi, gj = ghost
        if 0 <= gi < size and 0 <= gj < size and board[gi, gj] == 0:
            if to_play == 1:
                ax.add_artist(plt.Circle((gj, gi), 0.4, color="black", alpha=0.3, zorder=3))#type:ignore
            else:
                ax.add_artist(plt.Circle((gj, gi), 0.4, color="white", ec="black", alpha=0.3, zorder=3))#type:ignore

    # If game is over, show result, draw territiory
    if game_over and score is not None:
        for c, v in territory:
            if v == 1:  # black
                ax.add_artist(plt.Rectangle((c[0] - 0.2, c[1] - 0.2), 0.4, 0.4, color="black", alpha=0.6, zorder=2))#type:ignore
            elif v == -1:  # white
                ax.add_artist(plt.Rectangle((c[0] - 0.2, c[1] - 0.2), 0.4, 0.4, color="white", ec="black", alpha=0.6, zorder=2))#type:ignore

        result_text = f"Winner: {winner} | Score: {score}"
        ax.text(
            size / 2, size,
            result_text,
            fontsize=12,
            ha="center", va="bottom",
            color="red",
            fontweight="bold"
        )

    ax.set_xlim(-0.5, size-0.5)
    ax.set_ylim(-0.5, size-0.5)
    ax.axis("off")
    plt.draw()


def start_game_UI(saved_game=None, checkpoint_path=None, show_policy=False, play_bot=False, liveMCTS=False, args={}, player_first=True):
    if player_first:
        player_color = 1
    else:
        player_color = -1

    if liveMCTS:
        args['MCTS_SIMS'] = 99999
 
    size = args['BOARD_SIZE']
    max_game_len = args['MAX_GAME_LENGTH']
    board = np.zeros((size, size), dtype=np.int8)
    prev_board = None

    stop_flag = False
    last_ghost = None

    if saved_game:
        history = saved_game
    else:
        history = [board.copy()]
    history_index = 0
    to_play = 1
    passes = 0
    move_number = 0
    ghost = None
    game_over = False
    prev_move = None

    territory = []

    visit_counts = []
    pi = []
    MCTS_value = 0

    checkpoint = None
    if checkpoint_path:
        print(f"using checkpoint: {str(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model = AZNet9(board_size=BOARD_SIZE).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    plt.ion()
    plt.show(block=False)
    window_alive = True

    pass_ax = plt.axes([0.4, 0.05, 0.2, 0.075])#type:ignore
    pass_button = Button(pass_ax, "Pass")

    back_ax = plt.axes([0.2, 0.05, 0.2, 0.075])#type:ignore
    back_button = Button(back_ax, "back")

    forward_ax = plt.axes([0.6, 0.05, 0.2, 0.075])#type:ignore
    forward_button = Button(forward_ax, "forward")

    #########################################################################
    
    if checkpoint and (show_policy or liveMCTS):
        p, v = get_policy_value(model, history, to_play, BOARD_SIZE, DEVICE)
    else:
        p, v = None, None

    display_board(ax, board, history, history_index, prev_move=prev_move, policy=p if show_policy else None,
                   value_est=v if show_policy or liveMCTS else None, ghost=ghost, to_play=to_play,
                     territory=territory, play_bot=play_bot, player_move=not player_first)
    

    def end_game():
        nonlocal game_over, territory
        game_over = True
        score, territory = Game.final_score_tromp_taylor(board, size)
        winner = "Black" if score > 0 else "White" if score < 0 else "Tie"
        print(f"Game finished. Winner: {winner} (score {score})")

        # Redraw with final result
        display_board(ax, board, history, history_index, prev_move=prev_move, policy=p if show_policy else None,
                        value_est=v if show_policy or liveMCTS else None, ghost=ghost, to_play=to_play,
                          game_over=True, score=score, winner=winner, territory=territory, play_bot=play_bot)

    def make_move(move, player_move=True):
        nonlocal board, prev_board, history, history_index, to_play, passes, move_number, p, v, prev_move, stop_flag, visit_counts, pi, MCTS_value


        if Game.is_legal(board, move, to_play, prev_board, size):
            stop_flag = True

            move_number += 1
            prev_move = move
            new_board = Game.apply_move(board, move, to_play, size)

            if move is None:
                passes += 1
            else:
                passes = 0

            if passes >= 2 or move_number > max_game_len:
                end_game()
                return

            prev_board = board.copy()
            board = new_board
            history = history[:history_index+1] + [board.copy()] #########
            move_number = len(history) - 1
            history_index = len(history) - 1
            to_play = -to_play

            if checkpoint and (show_policy or liveMCTS):
                p, v = get_policy_value(model, history, to_play, BOARD_SIZE, DEVICE)

            ax.figure.canvas.draw_idle()
            plt.pause(0.05)

            display_board(ax, board, history, history_index, prev_move=prev_move, policy=p if show_policy else None,
                           value_est=v if show_policy or liveMCTS else None, ghost=ghost, to_play=to_play,
                             territory=territory, play_bot=play_bot, player_move=player_move)

            ax.figure.canvas.draw_idle()
            plt.pause(0.05)
        else:
            print("illegal move")
        
        if play_bot and to_play * player_color == -1: # bot turn
            mcts = MCTS(model, board, to_play, args, prev_board, debug=False)
            pi, _ = mcts.search(args['MIN_SEARCH_TIME'])

            # temperature schedule
            temp = get_temperature_for_move(move_number, args['TEMPERATURE_SCHEDULE'])
            pi = apply_temperature(pi, temp)

            # Select move
            action_idx = np.random.choice(len(pi), p=pi)
            action = Game.index_to_action(action_idx, args['BOARD_SIZE'])

            ax.figure.canvas.draw_idle()
            plt.pause(0.05)

            make_move(action, player_move=False)

            ax.figure.canvas.draw_idle()
            plt.pause(0.05)
        
        if liveMCTS and not game_over:
            run_live_mcts()


    def onclick(event):
        if event.inaxes != ax or game_over:
            return
        j, i = round(event.xdata), round(event.ydata)
        make_move((i, j))


    def onmotion(event):
        nonlocal ghost, last_ghost

        if liveMCTS:
            return

        if game_over or event.inaxes != ax:
            new_ghost = None
        else:
            j, i = round(event.xdata), round(event.ydata)
            new_ghost = (i, j)

        if new_ghost == last_ghost:
            return

        ghost = new_ghost
        last_ghost = new_ghost


    def onpass(event):
        nonlocal stop_flag
        if not game_over:
            stop_flag = True
            make_move(None)
        
    def onback(event):
        nonlocal history_index, history, board, move_number, to_play, p, v
        if history_index:
            move_number -= 1
            to_play = -to_play
            history_index -= 1
            board = history[history_index].copy()

            history_for_model = history[:history_index+1].copy()
            if checkpoint and (show_policy or liveMCTS):
                p, v = get_policy_value(model, history_for_model, to_play, BOARD_SIZE, DEVICE)

            display_board(ax, board, history, history_index, prev_move=prev_move, policy=p if show_policy else None,
                            value_est=v if show_policy or liveMCTS else None, ghost=ghost, to_play=to_play,
                              game_over=game_over, territory=territory, play_bot=play_bot)

    def onforward(event):
        nonlocal history_index, history, board, move_number, to_play, p, v
        if history_index + 1 < len(history):
            move_number += 1
            to_play = -to_play
            history_index += 1
            board = history[history_index].copy()

            history_for_model = history[:history_index+1].copy()
            if checkpoint and (show_policy or liveMCTS):
                p, v = get_policy_value(model, history_for_model, to_play, BOARD_SIZE, DEVICE)

            display_board(ax, board, history, history_index, prev_move=prev_move, policy=p if show_policy else None,
                            value_est=v if show_policy or liveMCTS else None, ghost=ghost, to_play=to_play,
                              game_over=game_over, territory=territory, play_bot=play_bot)

    def on_close(event):
        nonlocal window_alive, stop_flag
        window_alive = False
        stop_flag = True


    fig.canvas.mpl_connect("close_event", on_close)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("motion_notify_event", onmotion)
    pass_button.on_clicked(onpass)
    back_button.on_clicked(onback)
    forward_button.on_clicked(onforward)
    #################################################################################################################################################################

    if play_bot and to_play * player_color == -1: # bot turn
        ax.figure.canvas.draw_idle()
        plt.pause(0.05)
        
        mcts = MCTS(model, board, to_play, args, prev_board, debug=False)
        pi, _ = mcts.search(args['MIN_SEARCH_TIME'])

        temp = get_temperature_for_move(move_number, args['TEMPERATURE_SCHEDULE'])
        pi = apply_temperature(pi, temp)

        action_idx = np.random.choice(len(pi), p=pi)
        action = Game.index_to_action(action_idx, args['BOARD_SIZE'])

        make_move(action, False)

        ax.figure.canvas.draw_idle()
        plt.pause(0.05)

    def run_live_mcts():
        nonlocal stop_flag, visit_counts, pi, MCTS_value

        stop_flag = False

        mcts = MCTS_Go_Live.LiveMCTS(model, board, to_play, args, prev_board)
        for i, root in enumerate(mcts.itterate()):
            if not window_alive or stop_flag:
                break
            
            if i % 25 == 0:  # update every N sims
                visit_counts = np.zeros(args['BOARD_SIZE'] * args['BOARD_SIZE'] + 1, dtype=float)
                for child in root.children:
                    a = child.action_taken
                    idx = Game.action_to_index(a, args['BOARD_SIZE'])
                    visit_counts[idx] += child.visits

                pi = MCTS_Go_Live.get_probs(root, args, debug=False) # proportional to visit counts
                MCTS_value = MCTS_Go_Live.get_MCTS_value(root)#root.value/root.visits
                visit_counts = list(visit_counts)
                pi = list(pi)
                display_board(ax, board, history, history_index, prev_move=prev_move, policy=p if show_policy else None,
                               value_est=v if show_policy or liveMCTS else None, MCTS_visits=visit_counts, MCTS_probs=pi,
                                 MCTS_value=MCTS_value, ghost=ghost, to_play=to_play, territory=territory, play_bot=play_bot)
                
                ax.figure.canvas.draw_idle()
                plt.pause(0.05)

    if liveMCTS and not game_over:
        run_live_mcts()

    while window_alive:
        plt.pause(0.05)

#################################################################################################################################

if __name__ == "__main__":

    if mode == "sandbox":
        start_game_UI(args=args) # freeplay

    elif mode == "view policy":
        start_game_UI(show_policy=True, checkpoint_path=path, args=args) # policy

    elif mode == "live mcts":
        start_game_UI(saved_game=None, show_policy=False, checkpoint_path = path, play_bot=False, liveMCTS=True, args=args) #live mcts

    elif mode == "play game":
        start_game_UI(saved_game=None, show_policy=False, checkpoint_path=path, play_bot=True, args=args, player_first=player_goes_first) # play game

    else:
        print("mode not found")

