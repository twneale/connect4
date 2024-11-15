import os
import collections
import numpy as np
from scipy.signal import convolve2d


class GameState:
    '''Represents the state of the board at a given turn.
    '''
    X = 0
    # Diagonal kernels
    positive_diagonal_kernel = np.eye(4, dtype=int)  # Diagonal \
    negative_diagonal_kernel = np.fliplr(positive_diagonal_kernel)  # Diagonal /

    # Horizontal kernel
    horizontal_kernel = np.array([[1, 1, 1, 1]])

    # Vertical kernel
    vertical_kernel = np.array([[1], [1], [1], [1]])

    maxdepth = 4

    def __init__(self, parent=None, current_player=1, board=None, depth=0, maximize_player=1, abpruning=True):
        if board is None:
            board = np.zeros((6, 7), dtype=int)
        self.depth = depth
        self.board = board
        self.current_player = np.int64(current_player)
        self.maximize_player = np.int64(maximize_player)
        self.abpruning = abpruning
        self.parent = parent
        self.children = []
        self.winner = None

    def get_available_rows(self, board):
        """
        Returns the row index where a piece would drop in each column.
        If a column is full, it returns -1 for that column.
        """
        # Flip the board vertically so the lowest row becomes the first row in each column
        flipped_board = np.flipud(board == 0)

        # Find the first empty slot in each column (argmax gives the index of the first True value)
        first_empty_row = np.argmax(flipped_board, axis=0)

        # If the column is full, argmax will return 0, but we need to check if it’s truly full
        full_columns = (board[0, :] != 0)
        first_empty_row[full_columns] = -1  # Mark full columns as -1

        return first_empty_row

    def check_horizontal_win(self, board, player):
        # Apply the convolution
        result = convolve2d(board == player, self.horizontal_kernel, mode='valid')
        return np.any(result == 4)

    def check_vertical_win(self, board, player):
        # Apply the convolution
        result = convolve2d(board == player, self.vertical_kernel, mode='valid')
        return np.any(result == 4)

    def check_diagonal_win(self, board, player):
        # Positive diagonal (top-left to bottom-right)
        result_pos_diag = convolve2d(board == player, self.positive_diagonal_kernel, mode='valid')
        
        # Negative diagonal (top-right to bottom-left)
        result_neg_diag = convolve2d(board == player, self.negative_diagonal_kernel, mode='valid')
        
        return np.any(result_pos_diag == 4) or np.any(result_neg_diag == 4)

    def check_win(self):
        win = (
            self.check_horizontal_win(self.board, self.current_player) or
            self.check_vertical_win(self.board, self.current_player) or
            self.check_diagonal_win(self.board, self.current_player)
        )
        if win:
            self.winner = self.current_player
            #print('player %d wins at depth %d' % (self.current_player, self.depth))
            #print(self.board)
        return win

    def other_player(self):
        return {np.int64(1): np.int64(2), np.int64(2): np.int64(1)}[self.current_player]

    def describe_win(self):
        positive_diagonal_kernel = np.eye(4, dtype=int)  # Diagonal \
        negative_diagonal_kernel = np.fliplr(positive_diagonal_kernel)  # Diagonal /

        # Horizontal kernel
        horizontal_kernel = np.array([[1, 1, 1, 1]])

        # Vertical kernel
        vertical_kernel = np.array([[1], [1], [1], [1]])
        player = self.winner
        board = np.flipud(self.board)
        wins = []

        result = convolve2d(board == player, horizontal_kernel, mode='valid')
        for rows, cols in zip(*np.where(result == 4)):
            x = np.ones(4, dtype=int) * rows
            y = np.arange(cols, cols + horizontal_kernel.shape[1])
            coords = np.column_stack((x, y))
            wins.append(coords)

        result = convolve2d(board == player, vertical_kernel, mode='valid')
        for rows, cols in zip(*np.where(result == 4)):
            x = np.arange(vertical_kernel.shape[0]) + rows
            y = np.ones(vertical_kernel.shape[0], dtype=int) * cols
            coords = np.column_stack((x, y))
            wins.append(coords)

        result_pos_diag = convolve2d(board == player, positive_diagonal_kernel, mode='valid')
        for rows, cols in zip(*np.where(result_pos_diag == 4)):
            coords = np.column_stack((rows + np.arange(4), cols + np.arange(4)))
            wins.append(coords)

        # Negative diagonal (top-right to bottom-left)
        result_neg_diag = convolve2d(board == player, negative_diagonal_kernel, mode='valid')
        for rows, cols in zip(*np.where(result_neg_diag == 4)):
            coords = np.column_stack((rows + np.flip(np.arange(4)), cols + np.arange(4)))
            wins.append(coords)

        return wins

    def apply_move(self, move):
        col, row = move
        np.flipud(self.board)[row, col] = self.current_player
        self.move = (row, col, self.current_player)
        self.check_win()

    def new_child(self, move):
        child_board = self.board.copy()
        child = GameState(
          board=child_board, depth=self.depth + 1, 
          parent=self, current_player=self.other_player(),
          maximize_player=self.maximize_player,
          abpruning=self.abpruning
        )
        child.apply_move(move)
        self.children.append(child)
        return child

    def compute_moves(self):
        GameState.X += 1
        if self.winner is not None:
            return
        if self.depth >= self.maxdepth:
            return
        available_rows = self.get_available_rows(self.board)
        # Sort the moves with center-most moves at the front of the loop to help 
        # optimize alpha-beta pruning.
        colrows = list(sorted(enumerate(available_rows), key=lambda t: abs(t[0] - 3)))
        for col, row in colrows:
            if row == -1:
                continue
            child = self.new_child((col, row))
            child.compute_moves()
            child.score_position()

    def score_position(self):
        """
        Score the current board position for the given player by evaluating all possible
        4-piece windows in rows, columns, and diagonals.
        """
        score = 0
    
        # Score horizontal windows
        for row in range(6):
            for col in range(4):
                window = self.board[row, col:col+4]
                coords = np.column_stack((np.ones(4, dtype=int) * row, np.arange(col, col + 4)))
                score += self.evaluate_window(window, 'diag', coords, self.maximize_player)
        
        # Score vertical windows
        for col in range(7):
            for row in range(3):
                window = self.board[row:row+4, col]
                coords = np.column_stack((np.arange(row, row + 4), np.ones(4, dtype=int) * col))
                score += self.evaluate_window(window, 'vert', coords, self.maximize_player)
        
        # Score positive diagonal windows
        for row in range(3):
            for col in range(4):
                window = [self.board[row+i, col+i] for i in range(4)]
                coords = np.column_stack((np.arange(4, dtype=int) + row, np.arange(4, dtype=int) + col))
                score += self.evaluate_window(window, '+diag', coords, self.maximize_player)
        
        # Score negative diagonal windows
        for row in range(3, 6):
            for col in range(4):
                window = [self.board[row-i, col+i] for i in range(4)]
                coords = np.column_stack((row - np.arange(4, dtype=int), col - np.arange(4, dtype=int)))
                score += self.evaluate_window(window, '-diag', coords, self.maximize_player)
        
        self.score = score
        return score
          
    def evaluate_window(self, window, direction, coords, player):
        """
        Evaluate a 4-piece window for the given player.
        A window is a slice of 4 adjacent cells in a row, column, or diagonal.
        """
        score = 0
        opponent = np.int64(2) if player == np.int64(1) else np.int64(1)
        zero = np.int64(0) 
        if np.count_nonzero(window == player) == 4:
            score += float('inf')  # Win
        elif np.count_nonzero(window == player) == 3 and np.count_nonzero(window == zero) == 1:
            score += 100000  # Three pieces with one space
        elif np.count_nonzero(window == player) == 2 and np.count_nonzero(window == zero) == 2:
            score += 1000  # Two pieces with two spaces
        elif np.count_nonzero(window == player) == 1 and np.count_nonzero(window == zero) == 3:
            score += 10  # One piece with three spaces
        
        # Penalize if the window is more favorable to the opponent
        if np.count_nonzero(window == opponent) == 4:
            score -= float('inf')
        elif np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == zero) == 1:
            score -= 100000000000  # Block opponent's winning move
        elif np.count_nonzero(window == opponent) == 2 and np.count_nonzero(window == zero) == 2:
            score -= 1000  # Two pieces with two spaces
        elif np.count_nonzero(window == opponent) == 1 and np.count_nonzero(window == zero) == 3:
            score -= 10  # One piece with three spaces
        
        return score

    class BranchPruned(Exception):
        '''A janky substitute for break.
        '''

    def minimax_score(self, depth=0, alpha=-float('inf'), beta=float('inf')):
        GameState.X += 1
        if hasattr(self, 'mscore'):
            return self.mscore
        if depth >= self.maxdepth:
            return self.score

        available_rows = self.get_available_rows(self.board)
        # Sort the moves with center-most moves at the front of the loop to help 
        # optimize alpha-beta pruning.
        colrows = list(sorted(enumerate(available_rows), key=lambda t: abs(t[0] - 3)))

        if self.current_player != self.maximize_player:
            max_score = -float('inf')
            kids_by_score = collections.defaultdict(list)
            for i, (col, row) in enumerate(colrows):
                if row == -1:
                    continue
                child = self.new_child((col, row))
                child.score_position()
                score = child.minimax_score(alpha=alpha, beta=beta, depth=depth + 1)
                kids_by_score[score].append(child)
                max_score = max(max_score, score)
                if score == max_score:
                    # If multiple kids have the same minimax score, choose whichever has
                    # the lowest score for the next move, appearing to drag the game out
                    # as long as possible.
                    self.best_move = max(kids_by_score[max_score], key=lambda k: k.score)
                alpha = max(alpha, score)
                if self.abpruning and (alpha >= beta):
                    break
            score = max_score
        else:
            min_score = float('inf')
            kids_by_score = collections.defaultdict(list)
            for i, (col, row) in enumerate(colrows):
                if row == -1:
                    continue
                child = self.new_child((col, row))
                child.score_position()
                score = child.minimax_score(alpha=alpha, beta=beta, depth=depth + 1)
                kids_by_score[score].append(child)
                min_score = min(min_score, score)
                if score == min_score:
                    # If multiple kids have the same minimax score, choose the highest.
                    self.worst_move = max(kids_by_score[min_score], key=lambda k: k.score)
                beta = min(beta, score)
                if self.abpruning and (alpha >= beta):
                    break
            score = min_score
        self.mscore = score
        return score
 

board = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 2, 1, 2, 0],
                  [0, 0, 2, 1, 1, 1, 2]])


#x = score_position(board, player=np.int64(2))

#game = GameState(board=board, current_player=1, maximize_player=2, abpruning=True)
#game.score_position()
#game.minimax_score()
#print('move computed %d' % GameState.X)

def cow(node):
    for c in node.children:
        print(c.board)
        print(c.mscore)
        print(c.current_player)
#import pdb; pdb.set_trace()
#import pdb; pdb.set_trace()


#print("Player 1 Win?", check_win(board, 1))  # Check for Player 1
#print("Player 2 Win?", check_win(board, 2))  # Check for Player 2

