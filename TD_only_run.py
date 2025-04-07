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
    with open("train_file/td_learning_afterstate/td_table_episode_5000.pkl", "rb") as f:
        approximator.weights = pickle.load(f)

  
    state = env.reset()
    env.render()
    pre_score = 0

    done = False
    while not done:
 

        legal_moves = [a for a in range(4) if env.is_move_legal(a)]

        best_value = -float('inf')
        best_action = None

        for a in legal_moves:
            env_copy = copy.deepcopy(env)
            next_state, score_inc, done_flag, _, afterstate = env_copy.step(a)
            reward = score_inc - pre_score
            pre_score = score_inc
            v_after = approximator.value(afterstate)
            val = reward + v_after

            if val > best_value:
                best_value = val
                best_action = a
        state, reward, done, _, afterstate = env.step(best_action)
        print("best_act", best_action)
        print("reward", reward)


    print("Game over, final score:", env.score)

if __name__ == '__main__':
    main()


# def main():

    
#     env = Game2048Env()
#     patterns = get_patterns()
#     approximator = NTupleApproximator(board_size=4, patterns=patterns)
#     with open("train_file/td_learning/td_table_episode_5000.pkl", "rb") as f:
#         approximator.weights = pickle.load(f)

  
#     state = env.reset()
#     env.render()
#     pre_score = 0

#     done = False
#     while not done:
 

#         legal_moves = [a for a in range(4) if env.is_move_legal(a)]

#         best_value = -float('inf')
#         best_action = None

#         for a in legal_moves:
#             env_copy = copy.deepcopy(env)
#             next_state, score_inc, done_flag, _, afterstate = env_copy.step(a)
            
#             v_next = approximator.value(next_state)
#             if v_next > best_value:
#                 best_value = v_next
#                 best_action = a
#         state, reward, done, _, afterstate = env.step(best_action)
#         print("best_act", best_action)
#         print("reward", reward)


#     print("Game over, final score:", env.score)

# if __name__ == '__main__':
#     main()