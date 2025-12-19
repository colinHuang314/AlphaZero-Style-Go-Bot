import numpy as np
import math
import random
import time

from GoRules import Game  # has game rules
import BoardOperations_Go

import torch

class MCTSNode:
    def __init__(self, board, board_history, to_play, args, prev_board=None, action_taken=None, parent=None, prior=0.0):
       
        self.board = board
        self.to_play = to_play
        self.parent = parent
        self.prior = float(prior)
        self.prev_board = prev_board
        self.action_taken = action_taken

        self.args = args

        self.children = []  
        self.visits = 0.0
        self.value = 0.0  

        if board_history is None:
            self.board_history = [board]
        else:
            self.board_history = board_history

    def is_fully_expanded(self):
        return len(self.children) > 0

    def get_ucb(self, child):
        if child.visits == 0:
            q = 0.0
        else:
            q = - (child.value / child.visits) # flip for color flip

        u = self.args['CPUCT'] * child.prior * (math.sqrt(self.visits) / (1.0 + child.visits))
        return q + u

    def select(self):
        best_ucb = -np.inf
        best_children = []
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_children = [child]
                best_ucb = ucb
            elif ucb == best_ucb:
                best_children.append(child)
        return random.choice(best_children)

 
    def expand(self, policy, is_root):
       
        # Dirichlet noise
        if is_root:
            actions, probs = zip(*policy)  # separate into lists
            alpha = self.args['DIRICHLET_ALPHA']
            epsilon = self.args['DIRICHLET_EPSILON']

            if epsilon > 0 and alpha > 0: # 
                noise = np.random.dirichlet([alpha] * len(probs))
                probs = [(1 - epsilon) * p + epsilon * n for p, n in zip(probs, noise)]
                policy = list(zip(actions, probs))

        # normal expansion
        for action, prob in policy:
            if prob <= 0:
                continue            
           
            child_board = self.board.copy()
            child_board = Game.apply_move(child_board, action, self.to_play, self.args['BOARD_SIZE']) 

            child_history = self.board_history + [child_board]
            if len(child_history) > 8:  # keep last 8
                child_history = child_history[-8:]

            child = MCTSNode(child_board, child_history, self.to_play * -1, self.args, prev_board=self.board.copy(), action_taken=action, parent=self, prior=prob)
            self.children.append(child)


    def get_policy_value(self, model):
        model.eval()
        device = next(model.parameters()).device  # will be cpu during selfplay

        input = BoardOperations_Go.make_input_planes(
            self.board_history, self.to_play, self.args['BOARD_SIZE']
        )
        input = input.astype(np.float32)
        input = torch.tensor(input, dtype=torch.float32, device=device).unsqueeze(0)

        if input.ndim == 3:  
            input = input.unsqueeze(0)  


        with torch.no_grad():
            policy_logits, value = model(input)

        policy_logits = policy_logits.squeeze(0).cpu().numpy()  
        exp = np.exp(policy_logits - np.max(policy_logits))
        probs = exp / np.sum(exp)

        legal_mask = Game.get_legal_moves_mask(self.board, self.to_play, self.prev_board, self.args['BOARD_SIZE'])  # boolean array
        probs = probs * legal_mask

        probs = np.where(legal_mask > 0, np.clip(probs, self.args['MIN_PRIOR'], None), 0.0)

        probs /= probs.sum()
        

        size = self.args['BOARD_SIZE']
        policy = []
        for idx, prob in enumerate(probs):
            if idx == size * size:  # last index for pass move
                action = None
            else:
                r, c = divmod(idx, size)
                action = (r, c)
            policy.append((action, float(prob)))

        value = float(value.item())

        return policy, value


    def backpropagate(self, value):
        """
        value is from this node's perspective.
        We add value to this node, increment visits, and propagate -value to parent (since perspective flips).
        """
        self.value += value
        self.visits += 1.0
        if self.parent is not None:
            # parent sees the opposite perspective
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, model, board, to_play, args, prev_board=None, debug=False): # do we need action taken
        self.args = args
        self.model = model
        self.debug = debug
        self.board = board
        self.to_play = to_play
        self.prev_board = prev_board

    def search(self, min_run_time=0.0):
        root = MCTSNode(self.board.copy(), None, self.to_play, self.args, self.prev_board, None)

        size = self.args['BOARD_SIZE']
        sim = 1
        if min_run_time > 0: start = time.time()

        while sim <= int(self.args['MCTS_SIMS']) or (time.time() - start) < min_run_time:
            node = root
            passes = 0

            # descend
            while node.is_fully_expanded():
                node = node.select()
                if node.action_taken is None:
                    passes += 1
                else:
                    passes = 0

            is_terminal = (passes == 2)

            if not is_terminal:
                
                policy, value = node.get_policy_value(model=self.model)
                node.expand(policy, sim == 0)
            else:
                # terminal: score from black's perspective returned by Game.final_score_tromp_taylor
                raw_score, _ = Game.final_score_tromp_taylor(node.board, node.args['BOARD_SIZE'])
            
                z_black = 1.0 if raw_score > 0 else -1.0
                value = z_black * node.to_play

                '''
                certainty_bonus = 0.1 # experimental
                resign_bonus = 0.08
                if value > 0:
                    value += certainty_bonus
                else:
                    value += resign_bonus
                '''
                
            node.backpropagate(value)

            sim += 1
        ##########################################################################################################

        # build action_probs array
        action_probs = np.zeros(size * size + 1, dtype=float)

        for child in root.children:
            a = child.action_taken
            idx = Game.action_to_index(a, size)
            action_probs[idx] = child.visits

        # Normalize to sum = 1
        total = np.sum(action_probs)
        if total > 0:
            action_probs /= total

        if self.debug:
            print("--- MCTS Debug ---")
            for child in root.children:
                avg_child = child.value/child.visits if child.visits>0 else 0.0
                q_parent = -avg_child
                u = self.args['CPUCT'] * child.prior * (math.sqrt(root.visits) / (1 + child.visits))
                print(f"action: {child.action_taken}, visits: {child.visits}, prior: {child.prior:.3f}, avg_child: {avg_child:.3f}, Q_parent: {q_parent:.3f}, U: {u:.3f}")

        return action_probs, get_best_move_value(root)
    
def get_best_move_value(root):
    max_visits = 0
    best_child = None
    for child in root.children:
        if child.visits > max_visits:
            max_visits = child.visits
            best_child = child
    
    if best_child == None:
        return 0
    return - best_child.value / best_child.visits
