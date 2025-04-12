import copy
import random
import math
import numpy as np
from Game2048Env import Game2048Env, step_jit
from Otd_learning_Vinit_150000_lr02_numba import NTupleApproximator, convert_weights
from TD_MCTS import TD_MCTS, PlayerNode
import pickle
from n_tuple_design import get_patterns

def main():

    
    env = Game2048Env()
    patterns = get_patterns()
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    with open("train_file/OTD_learning_15W_lr02/td_table_episode_153000.pkl", "rb") as f:
        loaded_weights = pickle.load(f)
    approximator.weights = convert_weights(loaded_weights, approximator.patterns)


    iteration = 3
    total_score = 0

    for i in range(iteration):
        state = env.reset()

        done = False
        while not done:
            # Create the root node from the current state
            root = PlayerNode(state.copy(), env.score)
            td_mcts = TD_MCTS(env, approximator, iterations=200, exploration_constant=0, rollout_depth=1, gamma=0.95, Vnorm=220000)
            # Run multiple simulations to build the MCTS tree
            
            # print(root.state)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
            for _ in range(td_mcts.iterations):
                td_mcts.run_simulation(root)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

            # Select the best action (based on highest visit count)
            best_act, _ = td_mcts.best_action_distribution(root)

            # print(best_act)
            # Execute the selected action and update the state
            # state, reward, done, _, afterstate = env.step(best_act)
            # nxt_state, reward, done, _, afterstatee = step_jit(env.board.copy(), env.score, best_act)
            state, reward, done, _, afterstate = env.step(best_act)

            # state = nxt_state
            # env.board = state.copy()
            # env.score = reward

            # env.render(action=best_act)
            # print("TD-MCTS selected action:", best_act)
            # print("best_act", best_act)
            print("reward", reward)

        print(f"Game {i} over, final score:", env.score)
        total_score += env.score
    print(f"Evaluation finished, average score: {total_score/iteration}")

if __name__ == '__main__':
    main()