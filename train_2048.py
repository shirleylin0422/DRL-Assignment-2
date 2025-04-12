import copy
import random
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Game2048Env import Game2048Env, get_afterstate, is_move_legal_jit, step_jit
from tqdm import tqdm
import functools

from numba import njit
from numba.typed import Dict, List
from numba import types
# Otd_learning_Vinit_150000_lr05_numba

@njit
def tile_to_index_jit(tile):
    if tile == 0:
        return 0
    else:
        return int(math.log(tile) / math.log(2))

@njit
def get_feature_jit(board, coords):
    """
    board: numpy array 
    coords: numpy array，dtype=np.int64
    """
    return ( tile_to_index_jit(board[coords[0, 0], coords[0, 1]]),
             tile_to_index_jit(board[coords[1, 0], coords[1, 1]]),
             tile_to_index_jit(board[coords[2, 0], coords[2, 1]]),
             tile_to_index_jit(board[coords[3, 0], coords[3, 1]]),
             tile_to_index_jit(board[coords[4, 0], coords[4, 1]]),
             tile_to_index_jit(board[coords[5, 0], coords[5, 1]]) )

@njit
def value_jit(board, pattern_groups, weights, init_value):
    """
    board: numpy array
    pattern_groups: numba.typed.List
    weights: numba.typed.List
    """
    total_val = 0.0
    n_groups = len(pattern_groups)
    for g in range(n_groups):
        group = pattern_groups[g]   # group: typed List
        w_dict = weights[g]         # w_dict: numba.typed.Dict
        group_val = 0.0
        n_patterns = len(group)
        for p in range(n_patterns):
            coords = group[p]       
            feature = get_feature_jit(board, coords)
            if feature in w_dict:
                group_val += w_dict[feature]
            else:
                group_val += init_value
        total_val += group_val / n_patterns
    return total_val

@njit
def update_jit(board, pattern_groups, weights, delta, alpha, init_value):
    n_groups = len(pattern_groups)
    for g in range(n_groups):
        group = pattern_groups[g]
        w_dict = weights[g]
        n_patterns = len(group)
        for p in range(n_patterns):
            coords = group[p]
            feature = get_feature_jit(board, coords)
            if feature in w_dict:
                w_dict[feature] += alpha * delta / n_patterns
            else:
                w_dict[feature] = init_value + alpha * delta / n_patterns

@njit
def update_trajectory(trajectory, pattern_groups, weights, init_value, gamma, alpha):
    n_traj = len(trajectory)
    for idx in range(n_traj - 1, -1, -1):
        reward = trajectory[idx][0]
        next_v_after = trajectory[idx][1]
        afterstate = trajectory[idx][2] 
        v_s = value_jit(afterstate, pattern_groups, weights, init_value)
        td_error = reward + gamma * next_v_after - v_s
        update_jit(afterstate, pattern_groups, weights, td_error, alpha, init_value)

    return
class NTupleApproximator:
    def __init__(self, board_size, patterns):

        self.board_size = board_size
        # original n-tuple patterns pattern -> np.array (dtype=np.int64)
        self.patterns = tuple(np.array(pattern, dtype=np.int64) for pattern in patterns)
        V_init = 150000  
        num_patterns = len(self.patterns)
        self.init_value = V_init / num_patterns

        # weights -> numba.typed.List 
        # weight -> numba.typed.Dict
        weights_typed = List()
        for pattern in self.patterns:
            pattern_length = pattern.shape[0]  # 應該為 6
            weight_dict = Dict.empty(
                key_type=types.UniTuple(types.int64, pattern_length),
                value_type=types.float64
            )
            weights_typed.append(weight_dict)
        self.weights = weights_typed

        # store pattern_groups as numba.typed.List
        # pattern -> np.array, dtype=np.int64
        pattern_groups_typed = List()
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            group_typed = List()
            for s in syms:
                group_typed.append(np.array(s, dtype=np.int64))
            pattern_groups_typed.append(group_typed)
        self.pattern_groups = pattern_groups_typed

    def generate_symmetries(self, pattern):
        """
        pattern: np.array，(6,2)
        return list of tuple
        """
        # 將 np.array 轉成 list of tuple 以便計算
        pattern_list = [tuple(x) for x in pattern]
        added = set()
        syms = []
        def add_if_unique(p):
            key = tuple(p)
            if key not in added:
                added.add(key)
                syms.append(p)
        add_if_unique(pattern_list)
        rot90 = [(j, self.board_size - 1 - i) for (i, j) in pattern_list]
        add_if_unique(rot90)
        rot180 = [(self.board_size - 1 - i, self.board_size - 1 - j) for (i, j) in pattern_list]
        add_if_unique(rot180)
        rot270 = [(self.board_size - 1 - j, i) for (i, j) in pattern_list]
        add_if_unique(rot270)
        refl = [(i, self.board_size - 1 - j) for (i, j) in pattern_list]
        add_if_unique(refl)
        refl_rot90 = [(j, self.board_size - 1 - i) for (i, j) in refl]
        add_if_unique(refl_rot90)
        refl_rot180 = [(self.board_size - 1 - i, self.board_size - 1 - j) for (i, j) in refl]
        add_if_unique(refl_rot180)
        refl_rot270 = [(self.board_size - 1 - j, i) for (i, j) in refl]
        add_if_unique(refl_rot270)
        return syms

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))
    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[i, j]) for (i, j) in coords)

    def value(self, board):
        return value_jit(board, self.pattern_groups, self.weights, self.init_value)

    def update(self, board, delta, alpha):
        update_jit(board, self.pattern_groups, self.weights, delta, alpha, self.init_value)

@njit
def select_best_action_jit(board, env_score, previous_score, legal_moves,
                           pattern_groups, weights, init_value):

    best_value = -1e20
    best_action = -1
    n_moves = legal_moves.shape[0]
    for i in range(n_moves):
        a = legal_moves[i]
        # 注意：此處使用 jitted 版本的 get_afterstate
        new_state, new_score = get_afterstate(board.copy(), env_score, a)  # 盤面運算
        reward = new_score - previous_score
        v_after = value_jit(new_state, pattern_groups, weights, init_value)
        val = reward + v_after
        if val > best_value:
            best_value = val
            best_action = a
    return best_action, best_value
@njit
def select_next_best_jit(board, env_score, previous_score, legal_moves,
                           pattern_groups, weights, init_value):
    best_value = -1e20
    best_reward = 0.0
    best_v_after = 0.0
    n_moves = legal_moves.shape[0]
    for i in range(n_moves):
        a = legal_moves[i]
        new_state, new_score = get_afterstate(board.copy(), env_score, a)
        reward = new_score - previous_score
        v_after = value_jit(new_state, pattern_groups, weights, init_value)
        val = reward + v_after
        if val > best_value:
            best_value = val
            best_reward = reward
            best_v_after = v_after
    return best_reward, best_v_after

def convert_weights_for_pickle(typed_weights):
    py_list = []
    for d in typed_weights:
        py_dict = {}
        # 注意：使用 d.items() 可以遍歷每個 key, value
        for key, value in d.items():
            py_dict[key] = value
        py_list.append(py_dict)
    return py_list

def td_learning(env, approximator, start_episode=0, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    final_scores = []
    success_flags = []

    for episode in tqdm(range(start_episode, num_episodes), desc="Training Episodes"):
        if episode < num_episodes * 0.15:
            alpha = 0.5
        elif episode < num_episodes * 0.25:
            alpha = 0.3
        elif episode < num_episodes * 0.5:
            alpha = 0.2
        elif episode < num_episodes * 0.75:
            alpha = 0.15
        else:
            alpha = 0.1

        state = env.reset()
        trajectory = List()
        previous_score = env.score
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = np.array([a for a in range(4) if is_move_legal_jit(state, a)], dtype=np.int64)
            if legal_moves.size == 0:
                break

            best_action, best_value = select_best_action_jit(env.board.copy(), env.score, previous_score,
                                                     legal_moves, approximator.pattern_groups,
                                                     approximator.weights, approximator.init_value)
            action = best_action if best_action is not None else random.choice(legal_moves)
            
            # next_state, new_score, done, _, afterstate = env.step(action)
            next_state, new_score, done, _, afterstate = step_jit(env.board.copy(), env.score, action)
            env.board = next_state
            env.score = new_score
            reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            legal_moves = np.array([a for a in range(4) if is_move_legal_jit(next_state, a)], dtype=np.int64)
            if legal_moves.size == 0:
                break

            next_best_reward, next_best_v_after = select_next_best_jit(env.board.copy(), env.score, new_score,
                                                                legal_moves, approximator.pattern_groups,
                                                                approximator.weights, approximator.init_value)

            trajectory.append((next_best_reward, next_best_v_after, afterstate.copy()))
            state = next_state
        
        update_trajectory(trajectory, approximator.pattern_groups, approximator.weights, approximator.init_value, gamma, alpha)
        # for reward, next_v_after, afterstate in reversed(trajectory):
        #     v_s_ = approximator.value(afterstate)
        #     td_error = reward + gamma * next_v_after - v_s_
        #     approximator.update(afterstate, td_error, alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}", flush=True)
            if (episode + 1) % 1000 == 0:
                filename = f"train_file/OTD_learning_15W_lr05/td_table_episode_{episode+1}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(convert_weights_for_pickle(approximator.weights), f)
                print(f"Saved td-table to {filename}", flush=True)
        
    return final_scores

from n_tuple_design import get_patterns

def convert_weights(loaded_weights, patterns):
    """
    將 loaded_weights dict to numba.typed.Dict 
    """
    new_weights = List()
    for idx, d in enumerate(loaded_weights):
        pattern_length = patterns[idx].shape[0]
        new_d = Dict.empty(
            key_type=types.UniTuple(types.int64, pattern_length),
            value_type=types.float64
        )
        for key, value in d.items():
            new_d[tuple(key)] = value
        new_weights.append(new_d)
    return new_weights

def main():
    patterns = get_patterns() 
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    # with open("train_file/OTD_learning_15W_lr05/td_table_episode_19000.pkl", "rb") as f:
    #     loaded_weights = pickle.load(f)
    # approximator.weights = convert_weights(loaded_weights, approximator.patterns)
    
    env = Game2048Env()
    final_scores = td_learning(env, approximator, start_episode=0, num_episodes=153000, alpha=0.1, gamma=1)
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

if __name__ == '__main__':
    main()
