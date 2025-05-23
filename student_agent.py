# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import gc


from Game2048Env import Game2048Env
# from td_learning import NTupleApproximator
import pickle
from n_tuple_design import get_patterns
from Otd_learning_Vinit_150000_lr02_numba import NTupleApproximator, convert_weights
from TD_MCTS import TD_MCTS, PlayerNode


approximator = None
pre_score = 0
# approximator = NTupleApproximator(board_size=4, patterns=patterns)

def init_model():
    global approximator
    if approximator is None:
        gc.collect() 
        approximator = NTupleApproximator(board_size=4, patterns=get_patterns())
        path = "train_file/td_table_episode_153000.pkl"

        with open(path, "rb") as f:
            loaded_weights = pickle.load(f)
        approximator.weights = convert_weights(loaded_weights, approximator.patterns)


"""MCTS"""

def get_action(state, score):
    env = Game2048Env()
    env.board = state.copy() 
    env.score = score
    
    root = PlayerNode(state.copy(), env.score)
    init_model()
    td_mcts = TD_MCTS(env, approximator, iterations=200, exploration_constant=0, rollout_depth=1, gamma=0.95, Vnorm=220000)
    
    
    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)
    # print("TD-MCTS selected action:", best_act)
    # state, reward, done, _, afterstate = env.step(best_act)
    # # env.render(action=best_act)
    # print("best_act", best_act)
    # print("reward", reward)
    
    return best_act

"""TD approximator only"""

# def get_action(state, score):
#     init_model()
#     global pre_score

#     env = Game2048Env()
#     env.board = state.copy() 
#     env.score = score
    
#     legal_moves = [a for a in range(4) if env.is_move_legal(a)]

#     best_value = -float('inf')
#     best_action = None

#     for a in legal_moves:
#         env_copy = copy.deepcopy(env)
#         # next_state, score_inc, done_flag, _, afterstate = env_copy.step(a)
#         afterstate, score_inc = env_copy.get_afterstate(a)
#         reward = score_inc - score
#         v_after = approximator.value(afterstate)
#         val = reward + v_after
#         if val > best_value:
#             best_value = val
#             best_action = a

#     return best_action