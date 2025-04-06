import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from Game2048Env import Game2048Env

# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------




class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        # self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.pattern_groups = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.pattern_groups.append(syms)
        self.weights = [defaultdict(float) for _ in self.pattern_groups]

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.

        syms = []
        syms.append(tuple(sorted(pattern)))

        rot90 = [(j, self.board_size - 1 - i) for (i, j) in pattern]
        syms.append(tuple(sorted(rot90)))

        rot180 = [(self.board_size - 1 - i, self.board_size - 1 - j) for (i, j) in pattern]
        syms.append(tuple(sorted(rot180)))

        rot270 = [(self.board_size - 1 - j, i) for (i, j) in pattern]
        syms.append(tuple(sorted(rot270)))

        refl = [(i, self.board_size - 1 - j) for (i, j) in pattern]
        syms.append(tuple(sorted(refl)))

        refl_rot90 = [(j, self.board_size - 1 - i) for (i, j) in refl]
        syms.append(tuple(sorted(refl_rot90)))

        refl_rot180 = [(self.board_size - 1 - i, self.board_size - 1 - j) for (i, j) in refl]
        syms.append(tuple(sorted(refl_rot180)))

        refl_rot270 = [(self.board_size - 1 - j, i) for (i, j) in refl]
        syms.append(tuple(sorted(refl_rot270)))

        return list(set(syms))


    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[i, j]) for (i, j) in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        eval = 0.0
        for group, weight_dict in zip(self.pattern_groups, self.weights):
            group_value = 0.0
            for pattern in group:
                feature = self.get_feature(board, pattern)
                group_value += weight_dict[feature]
            group_value /= len(group)
            eval += group_value
        return eval

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for group, weight_dict in zip(self.pattern_groups, self.weights):
            for pattern in group:
                feature = self.get_feature(board, pattern)
                weight_dict[feature] += alpha * delta / len(group)

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)

        v_current = approximator.value(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            # if random.random() < epsilon:
            #     action = random.choice(legal_moves)
            # else:
            best_value = -float('inf')
            best_action = None
            for a in legal_moves:
                env_copy = copy.deepcopy(env)
                next_state, score_inc, done_flag, _ = env_copy.step(a)
                v_next = approximator.value(next_state)
                if v_next > best_value:
                    best_value = v_next
                    best_action = a
            action = best_action if best_action is not None else random.choice(legal_moves)



            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))


            # TODO: Store trajectory or just update depending on the implementation
            # v_next = approximator.value(next_state) if not done else 0.0

            # td_error = incremental_reward + gamma * v_next - v_current
            # approximator.update(state, td_error, alpha)


            # state = next_state
            # v_current = approximator.value(state)
            trajectory.append((state.copy(), incremental_reward))
            state = next_state

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
        v_next = 0.0
        for s, reward in reversed(trajectory):
            v_current = approximator.value(s)
            td_error = reward + gamma * v_next - v_current
            approximator.update(s, td_error, alpha)
            v_next = v_current

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 1000 == 0:
            filename = f"train_file/td_learning/td_table_episode_{episode+1}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(approximator.weights, f)
            print(f"Saved td-table to {filename}")

            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    return final_scores


# TODO: Define your own n-tuple patterns
# refer to Temporal Difference Learning of N-Tuple Networks for the Game 2048
from n_tuple_design import get_patterns


patterns = get_patterns()


approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=5000, alpha=0.05, gamma=0.99)
window = 100
moving_avg = [np.mean(final_scores[i:i+window]) for i in range(len(final_scores)-window+1)]
plt.figure(figsize=(10, 5))
plt.plot(range(window-1, len(final_scores)), moving_avg, label="Moving Average Score")
plt.xlabel("Episode")
plt.ylabel("Final Score")
plt.title("Training Progression: Episode vs. Final Score")
plt.legend()
plt.grid(True)
plt.show()