import copy
import random
import math
import numpy as np
from Game2048Env import Game2048Env
from td_learning import NTupleApproximator
from TD_MCTS import TD_MCTS, TD_MCTS_Node
import pickle
from n_tuple_design import get_patterns

def main():

    
    env = Game2048Env()
    patterns = get_patterns()
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    with open("train_file/td_learning/td_table_episode_5000.pkl", "rb") as f:
        approximator.weights = pickle.load(f)

    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

    state = env.reset()
    env.render()

    done = False
    while not done:
        # Create the root node from the current state
        root = TD_MCTS_Node(state, env.score)

        # Run multiple simulations to build the MCTS tree
        for _ in range(td_mcts.iterations):
            td_mcts.run_simulation(root)

        # Select the best action (based on highest visit count)
        best_act, _ = td_mcts.best_action_distribution(root)
        print("TD-MCTS selected action:", best_act)

        # Execute the selected action and update the state
        state, reward, done, _ = env.step(best_act)
        # env.render(action=best_act)
        print("best_act", best_act)
        print("reward", reward)

    print("Game over, final score:", env.score)

if __name__ == '__main__':
    main()