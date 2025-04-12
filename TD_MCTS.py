import copy
import random
import math
import numpy as np
from Game2048Env import Game2048Env, get_afterstate, is_move_legal_jit, step_jit

from numba import njit
import numpy as np
import math

@njit
def compute_best_child_action(parent_visits, rewards, visits, actions, approximator_values, exploration_constant):
    
    n = rewards.shape[0]

    # print("-------normalization--------")
    """min-max normalization"""
    q_values = np.empty(n, dtype=np.float64)
    for i in range(n):
        if visits[i] > 0:
            q_values[i] = rewards[i] / visits[i]
        else:
            q_values[i] = approximator_values[i]

    min_q = np.min(q_values)
    max_q = np.max(q_values)
    if max_q > min_q:
        q_norm_values = (q_values - min_q) / (max_q - min_q)
    else:
        q_norm_values = q_values  

    
    """compute best child action"""
    best_value = -1e20
    best_child_action = None

    for i in range(n):
        if visits[i] > 0:
            exploration = exploration_constant * math.sqrt(math.log(parent_visits) / visits[i])
        else:
            exploration = 0
        # print("q", q_norm_values[i])
        # print("exploration", exploration)
        uct = q_norm_values[i] + exploration
        if uct > best_value:
            best_value = uct
            best_child_action = actions[i]
    
    return best_child_action

@njit
def select_random_legal_move(board):

    legal = np.empty(4, dtype=np.int64)
    count = 0
    for a in range(4):
        if is_move_legal_jit(board, a):
            legal[count] = a
            count += 1
    if count == 0:
        return -1  
    idx = np.random.randint(0, count)
    return legal[idx]


@njit
def rollout_loop_jit(board, score, depth, gamma):

    rollout_reward = 0.0
    discount = 1.0
    current_board = board.copy()
    current_score = score 

    for d in range(depth):
        action = select_random_legal_move(current_board)
        if action == -1:
            break
        for a in range(4):
            if is_move_legal_jit(board, a):
                next_board, new_score, done, _, new_afterstate = step_jit(current_board, current_score, action)
                reward = new_score - current_score
                rollout_reward += discount * reward 
        discount *= gamma
        current_board = next_board.copy()
        current_score = new_score
        if done:
            break
    return rollout_reward, discount, new_afterstate

class PlayerNode:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves

        self.untried_actions = [a for a in range(4) if is_move_legal_jit(state,a)]


    def if_legal_actions(self):
        legal_actions = []
        for a in range(4) :
            if is_move_legal_jit(self.state,a):
                legal_actions.append(a)
        
        if legal_actions != []:
            return True, legal_actions
        return False, []
            
    
    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        if_legal_action, legal_actions = self.if_legal_actions()
        if not if_legal_action:
            return False
        return all(action in self.children for action in legal_actions)

class ChanceNode:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
       
        self.expanded = False

    
    def fully_expanded(self):
        return self.expanded


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99, Vnorm=1):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.Vnorm = Vnorm

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        n = len(node.children)
        rewards = np.empty(n, dtype=np.float64)
        visits = np.empty(n, dtype=np.float64)
        actions = np.empty(n, dtype=np.int64)
        approximator_values = np.empty(n, dtype=np.int64)
        i = 0
        for action, child in node.children.items():
            rewards[i] = child.total_reward
            visits[i] = child.visits
            actions[i] = action
            if visits[i] == 0:
                approximator_values[i] = self.approximator.value(child.state)
            else:
                approximator_values[i] = 0
            i += 1
        parent_visits = node.visits
        best_action = compute_best_child_action(parent_visits, rewards, visits, actions, approximator_values, self.c)
        
        return best_action


    def rollout(self, node, rollout_depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.

        if isinstance(node, PlayerNode):

            """approximator rollout"""
            best_value = float('-inf')
            has_legal_mv, legal_actions = node.if_legal_actions()
            rollout_reward = 0
            if has_legal_mv:
                for a in legal_actions:
                    next_board, new_score, done, _, new_afterstate = step_jit(node.state.copy(), node.score, a)
                    reward = new_score - node.score
                    val = reward + self.approximator.value(new_afterstate)
                    if val > best_value:    
                        best_value = val
                    
                rollout_reward = best_value
            
            """random rollout"""
            # sim_env = self.create_env_from_state(node.state.copy(), node.score)
            # rollout_reward = 0
            # for i in range(rollout_depth):
            #     legal_moves = [a for a in range(4) if is_move_legal_jit(sim_env.board, a)]
            #     if not legal_moves:
            #         break

            #     a = random.choice(legal_moves)
            #     new_state, new_score, done, _, new_afterstate = step_jit(node.state.copy(), node.score, a)
            #     reward = new_score - sim_env.score
            #     rollout_reward += reward + self.approximator.value(new_afterstate)
            #     sim_env = self.create_env_from_state(new_state, new_score)
            #     if done:
            #         break
                
            



        elif isinstance(node, ChanceNode):
            rollout_reward = self.approximator.value(node.state)

    
        return rollout_reward

    def backpropagate(self, node, rollout_reward, selection_rewards):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.

        reward_idx = len(selection_rewards) - 1
        # last node is chancenode
        if isinstance(node, ChanceNode):
            node.total_reward += rollout_reward
            node.visits += 1
            node = node.parent
            while node is not None:
                node.visits += 1
                if isinstance(node, ChanceNode):
                    reward = selection_rewards[reward_idx]
                    rollout_reward = rollout_reward + reward
                    node.total_reward += rollout_reward
                node = node.parent
                reward_idx -= 1
        elif isinstance(node, PlayerNode):
            node.visits += 1
            node = node.parent
            if node is not None and isinstance(node, ChanceNode):
                node.total_reward += rollout_reward
                node.visits += 1
                node = node.parent
            while node is not None:
                node.visits += 1
                if isinstance(node, ChanceNode):
                    reward = selection_rewards[reward_idx]
                    rollout_reward = rollout_reward + reward
                    node.total_reward += rollout_reward
                node = node.parent
                reward_idx -= 1

    
    def run_simulation(self, root):
        node = root
        
        

        """Selection: Traverse the tree until reaching a non-fully expanded node."""
        selection_rewards = []
        # print("---------------Selection-----------------")
        # print(node.state)
        # print(f"fully_expanded {node.fully_expanded()}")
        while node.fully_expanded(): 
            
            if isinstance(node, PlayerNode):
                best_action = self.select_child(node)
                _, new_score, _, _, new_afterstate = step_jit(node.state.copy(), node.score, best_action)
                reward = new_score - node.score
                selection_rewards.append(reward)
                # print("PlayerNode state after action")
                node = node.children[best_action]
            elif isinstance(node, ChanceNode):
                keys = list(node.children.keys())
                if len(keys) == 0:
                    break 
                weights = []
                for key in keys:
                    pos, tile = key
                    weight = 0.9 if tile == 2 else 0.1
                    weights.append(weight)
                chosen_key = random.choices(keys, weights=weights, k=1)[0]
                node = node.children[chosen_key]
        # print(node.state, node.score)

        """Expansion: if the node has untried actions, expand one."""
       
        # print("---------------Expansion-----------------")
        # sim_env = self.create_env_from_state(node.state, node.score)
        if isinstance(node, PlayerNode):
            # print("PlayerNode Expansion")
            if not node.children:
                for a in range(4):
                    # print(f"actopn {a} move legal? {is_move_legal_jit(node.state.copy(), a)}")
                    if is_move_legal_jit(node.state.copy(), a):
                        # print("move legal!!!!!!!!!!!!!!!!!!!")
                        _, new_score, _, _, afterstate = step_jit(node.state.copy(), node.score, a)
            
                        child = ChanceNode(afterstate.copy(), new_score, parent=node, action=a)
                        node.children[a] = child
                        # print("PlayerNode -> expanded ChanceNode")
                        # print(node.children[a].state, node.children[a].score)
        elif isinstance(node, ChanceNode):
            # print("ChanceNode Expansion")
            if not node.expanded:
                empty_positions = list(zip(*np.where(node.state == 0)))
                for pos in empty_positions:
                    for tile_value in [2, 4]:
                        new_state = node.state.copy()
                        new_state[pos] = tile_value
                        candidate_key = (pos, tile_value)
                        if candidate_key not in node.children:
                            child = PlayerNode(new_state, node.score, parent=node, action=candidate_key)
                            node.children[candidate_key] = child
                        # print("ChanceNode -> expanded PlayerNode")
                        # print(node.children[candidate_key].state, node.children[candidate_key].score)
                               
                node.expanded = True
        # print("Node after expanded")
        # print(node.state, node.score)
        # print("---------------Rollout-----------------")
        """Rollout: Simulate a random game from the expanded node."""
        rollout_reward = self.rollout(node, self.rollout_depth)
        # print("Node after rollout")
        # print(node.state, node.score)

        # print("---------------Backpropagate-----------------")
        """Backpropagate the obtained reward."""
        self.backpropagate(node, rollout_reward, selection_rewards)



    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


