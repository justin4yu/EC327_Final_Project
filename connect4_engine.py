# connect4_engine.py
import numpy as np
import pygame
import sys
import math
from agent import RLAgent

# Colors for UI
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)      # Human
YELLOW = (255, 255, 0) # AI Agent

ROWS = 6
COLS = 7
SQUARESIZE = 100

class Connect4Env:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS))
        self.game_over = False

    def reset(self):
        self.board = np.zeros((ROWS, COLS))
        self.game_over = False
        return self.board

    def is_valid_location(self, col):
        return self.board[ROWS-1][col] == 0

    def get_valid_locations(self):
        return [col for col in range(COLS) if self.is_valid_location(col)]

    def get_next_open_row(self, col):
        for r in range(ROWS):
            if self.board[r][col] == 0:
                return r

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def winning_move(self, piece, board=None):
        if board is None:
            board = self.board
        # Horizontal
        for c in range(COLS-3):
            for r in range(ROWS):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True
        # Vertical
        for c in range(COLS):
            for r in range(ROWS-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True
        # Positive Diagonal
        for c in range(COLS-3):
            for r in range(ROWS-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True
        # Negative Diagonal
        for c in range(COLS-3):
            for r in range(3, ROWS):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        return False

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = 1 if piece == 2 else 2

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 50 # Strongly penalize ignoring opponent threats

        return score

    def get_reward_and_features(self, piece):
        """Calculates shaped reward and extracts 'Mario Kart' style features"""
        score = 0
        
        # Feature 1: Center Control
        center_array = [int(i) for i in list(self.board[:, COLS//2])]
        center_count = center_array.count(piece)
        score += center_count * 2

        # Evaluate all windows for 3-in-a-rows (Threats)
        for r in range(ROWS):
            row_array = [int(i) for i in list(self.board[r,:])]
            for c in range(COLS-3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, piece)

        for c in range(COLS):
            col_array = [int(i) for i in list(self.board[:,c])]
            for r in range(ROWS-3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, piece)

        features = {
            "center_pieces": center_count,
            "heuristic_score": score
        }
        return score, features

class GameUI:
    def __init__(self, env, agent):
        pygame.init()
        self.env = env
        self.agent = agent
        self.width = COLS * SQUARESIZE
        self.height = (ROWS + 1) * SQUARESIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.radius = int(SQUARESIZE / 2 - 5)
        pygame.display.set_caption("RL Connect 4 Testbed")

    def draw_board(self):
        for c in range(COLS):
            for r in range(ROWS):
                pygame.draw.rect(self.screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(self.screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), self.radius)
        
        for c in range(COLS):
            for r in range(ROWS):
                if self.env.board[r][c] == 1:
                    pygame.draw.circle(self.screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), self.height - int(r*SQUARESIZE+SQUARESIZE/2)), self.radius)
                elif self.env.board[r][c] == 2:
                    pygame.draw.circle(self.screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), self.height - int(r*SQUARESIZE+SQUARESIZE/2)), self.radius)
        pygame.display.update()

    def play(self):
        self.draw_board()
        turn = 0 # 0 = Human, 1 = Agent
        font = pygame.font.SysFont("monospace", 75)

        while not self.env.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                # HUMAN TURN
                if event.type == pygame.MOUSEBUTTONDOWN and turn == 0:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if self.env.is_valid_location(col):
                        row = self.env.get_next_open_row(col)
                        self.env.drop_piece(row, col, 1)

                        if self.env.winning_move(1):
                            label = font.render("Human wins!!", 1, RED)
                            self.screen.blit(label, (40, 10))
                            self.env.game_over = True

                        turn += 1
                        turn = turn % 2
                        self.draw_board()

            # AGENT TURN
            if turn == 1 and not self.env.game_over:
                # 1. Get valid moves & features
                valid_moves = self.env.get_valid_locations()
                _, features = self.env.get_reward_and_features(piece=2)
                
                # 2. Ask your RL Agent for a move
                col = self.agent.get_action(self.env.board.copy(), valid_moves, features)
                
                if self.env.is_valid_location(col):
                    pygame.time.wait(500) # Slight delay so you can see the move
                    row = self.env.get_next_open_row(col)
                    self.env.drop_piece(row, col, 2)

                    if self.env.winning_move(2):
                        label = font.render("Agent wins!!", 1, YELLOW)
                        self.screen.blit(label, (40, 10))
                        self.env.game_over = True

                    turn += 1
                    turn = turn % 2
                    self.draw_board()

        pygame.time.wait(3000)

if __name__ == "__main__":
    env = Connect4Env()
    my_agent = RLAgent()
    
    # Start Interactive Play
    ui = GameUI(env, my_agent)
    ui.play()