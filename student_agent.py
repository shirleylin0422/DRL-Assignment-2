# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math


from Game2048Env import Game2048Env
from td_learning import NTupleApproximator
import pickle
from n_tuple_design import get_patterns
from TD_MCTS import TD_MCTS, TD_MCTS_Node

patterns = get_patterns()
approximator = NTupleApproximator(board_size=4, patterns=patterns)
with open("train_file/td_learning/td_table_episode_1000.pkl", "rb") as f:
    approximator.weights = pickle.load(f)


def get_action(state, score):
    env = Game2048Env()
    env.board = state.copy() 
    env.score = score
    
    root = TD_MCTS_Node(state, score)
    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)
    
    
    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)
    
    return best_act

    # return random.choice([0, 1, 2, 3]) # Choose a random action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


