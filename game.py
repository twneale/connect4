import time
import math
import random
import itertools

import pygame
import numpy as np

import ai

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = SQUARESIZE // 2 - 5
WIDTH = COLUMN_COUNT * SQUARESIZE
HEIGHT = (ROW_COUNT + 1) * SQUARESIZE
SCREEN_SIZE = (WIDTH, HEIGHT)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GOLD = (255, 215, 0)
BRONZE = (205, 127, 50)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Connect 4")

# Function to create the board (matrix representation)
def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    if random.random() > 0.5:
        board[0,3] = 2
    return board

def draw_star(screen, color, center, radius):
    # A star has 5 points, so 10 points for both the outer and inner vertices
    points = []
    for i in range(10):
        # Alternate between the outer and inner points
        angle = math.pi / 5 * i
        dist = radius if i % 2 == 0 else radius // 2
        x = center[0] + int(math.cos(angle) * dist)
        y = center[1] - int(math.sin(angle) * dist)
        points.append((x, y))

    # Draw the star using the points list
    pygame.draw.polygon(screen, color, points)

# Function to draw the Connect 4 board
def draw_board(board, wins=None):
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT):
            pygame.draw.rect(screen, BLUE, (col * SQUARESIZE, (row + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (col * SQUARESIZE + SQUARESIZE // 2, (row + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)

    win_coords = list(map(list, itertools.chain(*(wins or []))))
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT):
            if board[row][col] == 1:
                pygame.draw.circle(screen, RED, (col * SQUARESIZE + SQUARESIZE // 2, HEIGHT - (row + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)
                if [row, col] in win_coords:
                    center_x = col * SQUARESIZE + SQUARESIZE // 2
                    center_y = HEIGHT - (row + 1) * SQUARESIZE + SQUARESIZE // 2
                    draw_star(screen, GOLD, (center_x, center_y), RADIUS)
            elif board[row][col] == 2:
                color = YELLOW
                pygame.draw.circle(screen, color, (col * SQUARESIZE + SQUARESIZE // 2, HEIGHT - (row + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)
                if [row, col] in win_coords:
                    center_x = col * SQUARESIZE + SQUARESIZE // 2
                    center_y = HEIGHT - (row + 1) * SQUARESIZE + SQUARESIZE // 2
                    draw_star(screen, BRONZE, (center_x, center_y), RADIUS)
 

    pygame.display.update()

# Function to drop a piece in the grid
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Function to check if the column is valid for dropping a piece
def is_valid_column(board, col):
    return board[ROW_COUNT - 1][col] == 0

# Function to get the next open row in a column
def get_next_open_row(board, col):
    for row in range(ROW_COUNT):
        if board[row][col] == 0:
            return row


def play_game():
    # Main game loop
    board = create_board()
    draw_board(board)

    game_over = False

    while not game_over:
        for event in pygame.event.get():
            wins = []
            if event.type == pygame.QUIT:
                game_over = True
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    import pdb; pdb.set_trace()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
                posx = event.pos[0]
                pygame.draw.circle(screen, RED, (posx, SQUARESIZE // 2), RADIUS)
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # Player 1's turn (red)
                posx = event.pos[0]
                col = posx // SQUARESIZE

                if is_valid_column(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, 1)
                draw_board(board)

                # Model the board.
                state = ai.GameState(board=np.flipud(board), maximize_player=2)
                # Check if player 1 won the match.
                if state.check_win():
                    game_over = True
                    wins = state.describe_win()
                    draw_board(board, wins=wins)
                    break
                # Player 2's turn (Yellow)
                state.score_position()
                state.minimax_score()
                if state.best_move.check_win():
                    game_over = True
                    wins = state.best_move.describe_win()
                board = np.flipud(state.best_move.board)
                draw_board(board, wins=wins)
 
    time.sleep(3)

while True:
    play_game()
