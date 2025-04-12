import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# def compress(row, size=4):
#     """Compress the row: move non-zero values to the left"""
#     new_row = row[row != 0]  # Remove zeros
#     new_row = np.pad(new_row, (0, size - len(new_row)), mode='constant')  # Pad with zeros on the right
#     return new_row
@njit
def arrays_equal(arr1, arr2):
    if arr1.shape[0] != arr2.shape[0]:
        return False
    for i in range(arr1.shape[0]):
        if arr1[i] != arr2[i]:
            return False
    return True

@njit
def compress(row, size=4):
    count = 0
    # 計算非零值個數
    for i in range(size):
        if row[i] != 0:
            count += 1
    new_row = np.empty(size, dtype=row.dtype)
    idx = 0
    for i in range(size):
        if row[i] != 0:
            new_row[idx] = row[i]
            idx += 1
    # 補零
    for i in range(idx, size):
        new_row[i] = 0
    return new_row
    
@njit
def merge(row, score, size=4):
    for i in range(size - 1):
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] = row[i] * 2
            row[i + 1] = 0
            score += row[i]
    return row, score

@njit
def move_left(state, score, size=4):
    moved = False
    for i in range(size):
        original_row = state[i].copy()
        row = state[i].copy() 
        row = compress(row, size)
        row, score = merge(row, score, size)
        row = compress(row, size)
        state[i] = row
        if not arrays_equal(original_row, state[i]):
            moved = True
    return state, score, moved

@njit
def move_right(state, score, size=4):
    moved = False
    for i in range(size):
        original_row = state[i].copy()
        row = state[i].copy()
        rev = np.empty(size, dtype=row.dtype)
        for j in range(size):
            rev[j] = row[size - 1 - j]
        rev = compress(rev, size)
        rev, score = merge(rev, score, size)
        rev = compress(rev, size)
        new_row = np.empty(size, dtype=row.dtype)
        for j in range(size):
            new_row[j] = rev[size - 1 - j]
        state[i] = new_row
        if not arrays_equal(original_row, state[i]):
            moved = True
    return state, score, moved

@njit
def move_up(state, score, size=4):
    moved = False
    for j in range(size):
        col = np.empty(size, dtype=state.dtype)
        original_col = np.empty(size, dtype=state.dtype)
        for i in range(size):
            original_col[i] = state[i, j]
            col[i] = state[i, j]
        col = compress(col, size)
        col, score = merge(col, score, size)
        col = compress(col, size)
        for i in range(size):
            state[i, j] = col[i]
        for i in range(size):
            if state[i, j] != original_col[i]:
                moved = True
                break
    return state, score, moved

@njit
def move_down(state, score, size=4):
    moved = False
    for j in range(size):
        col = np.empty(size, dtype=state.dtype)
        original_col = np.empty(size, dtype=state.dtype)
        for i in range(size):
            original_col[i] = state[i, j]
            col[i] = state[i, j]
        rev = np.empty(size, dtype=col.dtype)
        for i in range(size):
            rev[i] = col[size - 1 - i]
        rev = compress(rev, size)
        rev, score = merge(rev, score, size)
        rev = compress(rev, size)
        new_col = np.empty(size, dtype=col.dtype)
        for i in range(size):
            new_col[i] = rev[size - 1 - i]
        for i in range(size):
            state[i, j] = new_col[i]
        for i in range(size):
            if state[i, j] != original_col[i]:
                moved = True
                break
    return state, score, moved


@njit
def get_afterstate(state, score, action):
    if action == 0:
        state, new_score, _ = move_up(state, score)
    elif action == 1:
        state, new_score, _ = move_down(state, score)
    elif action == 2:
        state, new_score, _ = move_left(state, score)
    elif action == 3:
        state, new_score, _ = move_right(state, score)

    return state, new_score

@njit
def add_random_tile(board):
    """Add a random tile (2 or 4) to an empty cell"""
    x_indices, y_indices = np.where(board == 0)
    n_empty = x_indices.shape[0]
    if n_empty > 0:
        idx = np.random.randint(0, n_empty)
        x = x_indices[idx]
        y = y_indices[idx]
        if np.random.random() < 0.9:
            board[x, y] = 2
        else:
            board[x, y] = 4
    return board

@njit
def is_game_over_jit(board, size=4):
    if np.any(board == 0):
        return False

    for i in range(size):
        for j in range(size - 1):
            if board[i, j] == board[i, j+1]:
                return False

    for j in range(size):
        for i in range(size - 1):
            if board[i, j] == board[i+1, j]:
                return False

    return True

@njit
def step_jit(state, score, action):
        """Execute one action"""

        if action == 0:
            state, new_score, moved = move_up(state, score)
        elif action == 1:
            state, new_score, moved  = move_down(state, score)
        elif action == 2:
            state, new_score, moved  = move_left(state, score)
        elif action == 3:
            state, new_score, moved  = move_right(state, score)
        else:
            moved = False

        afterstate = state.copy()

        if moved:
            state = add_random_tile(state)

        done = is_game_over_jit(state)

        return state, new_score, done, 0, afterstate

@njit
def simulate_row_move_jit(row, size=4):
    new_row = np.empty(size, dtype=row.dtype)
    count = 0
    for i in range(size):
        if row[i] != 0:
            new_row[count] = row[i]
            count += 1
    for i in range(count, size):
        new_row[i] = 0

    for i in range(size - 1):
        if new_row[i] == new_row[i + 1] and new_row[i] != 0:
            new_row[i] = new_row[i] * 2
            new_row[i + 1] = 0
    final_row = np.empty(size, dtype=row.dtype)
    count = 0
    for i in range(size):
        if new_row[i] != 0:
            final_row[count] = new_row[i]
            count += 1
    for i in range(count, size):
        final_row[i] = 0

    return final_row

@njit
def is_move_legal_jit(board, action, size=4):
   
    temp_board = board.copy()
    if action == 0:  
        for j in range(size):
            col = board[:, j]
            new_col = simulate_row_move_jit(col, size)
            temp_board[:, j] = new_col
    elif action == 1: 
        for j in range(size):
            col = board[:, j][::-1]
            new_col = simulate_row_move_jit(col, size)
            temp_board[:, j] = new_col[::-1]
    elif action == 2: 
        for i in range(size):
            row = board[i]
            new_row = simulate_row_move_jit(row, size)
            temp_board[i] = new_row
    elif action == 3: 
        for i in range(size):
            row = board[i][::-1]
            new_row = simulate_row_move_jit(row, size)
            temp_board[i] = new_row[::-1]
    else:
        return False
    return not np.array_equal(board, temp_board)

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def get_afterstate(self, action):
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        return self.board, self.score


    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        afterstate = self.board.copy()

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}, afterstate


    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)
