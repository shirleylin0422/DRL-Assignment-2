import copy
import random
import math
import numpy as np
from Game2048Env import Game2048Env

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        env = Game2048Env()
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            q_value = child.total_reward / child.visits if child.visits > 0 else 0
            # UCT
            uct_value = q_value + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if uct_value > best_score:
                best_score = uct_value
                best_child = child
        return best_child

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        cumulative_reward = 0.0
        current_depth = 0
        done = False
        while current_depth < depth and not done:
            legal_moves = [action for action in range(4) if sim_env.is_move_legal(action)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            state, reward, done, _ = sim_env.step(action)
            cumulative_reward += reward
            current_depth += 1

        cumulative_reward += (self.gamma ** current_depth) * self.approximator.value(sim_env.board)

        return cumulative_reward

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            _, reward, done, _ = sim_env.step(node.action)
            if done:
                break

        # TODO: Expansion: if the node has untried actions, expand one.
        if node.untried_actions:
          action = random.choice(node.untried_actions)
          node.untried_actions.remove(action)
          state, reward, done, _ = sim_env.step(action)
          child = TD_MCTS_Node(state, sim_env.score, parent=node, action=action)
          node.children[action] = child
          node = child


        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

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


