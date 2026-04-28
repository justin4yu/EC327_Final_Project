# connect4_engine.py
import numpy as np
import pygame
import sys
import math
import os
import random
import tkinter as tk
from tkinter import filedialog

# Dynamically import both agents and print the exact errors if they fail
try:
    from q_agent import RLAgent as QAgent
    print("SUCCESS: q_agent.py loaded perfectly.")
except Exception as e:
    print(f"CRITICAL ERROR loading q_agent.py: {e}")
    QAgent = None

try:
    from cnn_agent import RLAgent as CNNAgent
    print("SUCCESS: cnn_agent.py loaded perfectly.")
except Exception as e:
    print(f"CRITICAL ERROR loading cnn_agent.py: {e}")
    CNNAgent = None

# Colors for UI
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)      
YELLOW = (255, 255, 0) 
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
GREEN = (0, 200, 0)

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
        for c in range(COLS-3):
            for r in range(ROWS):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True
        for c in range(COLS):
            for r in range(ROWS-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True
        for c in range(COLS-3):
            for r in range(ROWS-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True
        for c in range(COLS-3):
            for r in range(3, ROWS):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        return False

class GameUI:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.width = COLS * SQUARESIZE
        self.height = (ROWS + 1) * SQUARESIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.radius = int(SQUARESIZE / 2 - 5)
        pygame.display.set_caption("Connect 4 AI Arena")
        self.font = pygame.font.SysFont("monospace", 50, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 25, bold=True)
        
        # Menu State
        self.human_first = True
        self.model_path = None
        self.agent = None
        self.model_name = "Random Moves (No Model)"

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw() 
        root.attributes('-topmost', True) 
        filepath = filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=(("PyTorch CNN", "*.pth"), ("Q-Table", "*.pkl"), ("All files", "*.*"))
        )
        root.destroy()
        return filepath

    def load_model(self, filepath):
        if not filepath:
            return
        
        ext = os.path.splitext(filepath)[1].lower()
        self.model_path = filepath
        self.model_name = os.path.basename(filepath)

        if ext == '.pkl' and QAgent:
            self.agent = QAgent(epsilon=0.0)
            self.agent.load(filepath)
            print(f"Loaded Q-Table: {self.model_name}")
        elif ext == '.pth' and CNNAgent:
            self.agent = CNNAgent(epsilon=0.0)
            self.agent.load(filepath)
            print(f"Loaded CNN: {self.model_name}")
        else:
            print("Failed to load. Unknown file type or missing agent file.")
            self.agent = None
            self.model_name = "Load Failed - Playing Randomly"

    def draw_text(self, text, font, color, y_offset):
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=(self.width/2, y_offset))
        self.screen.blit(surface, rect)

    def main_menu(self):
        menu_running = True
        
        # Define button rects
        btn_width = 350
        btn_height = 60
        btn_x = (self.width - btn_width) // 2
        
        turn_btn = pygame.Rect(btn_x, 250, btn_width, btn_height)
        model_btn = pygame.Rect(btn_x, 350, btn_width, btn_height)
        start_btn = pygame.Rect(btn_x, 500, btn_width, btn_height)

        while menu_running:
            self.screen.fill(BLACK)
            
            self.draw_text("CONNECT 4 SETUP", self.font, WHITE, 100)
            self.draw_text(f"Current Model: {self.model_name}", self.small_font, YELLOW, 170)

            # Draw Buttons
            pygame.draw.rect(self.screen, GRAY, turn_btn)
            pygame.draw.rect(self.screen, GRAY, model_btn)
            pygame.draw.rect(self.screen, GREEN, start_btn)

            turn_text = "Turn: Human First" if self.human_first else "Turn: AI First"
            self.draw_text(turn_text, self.small_font, BLACK, 280)
            self.draw_text("Select Model File", self.small_font, BLACK, 380)
            self.draw_text("START GAME", self.small_font, BLACK, 530)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if turn_btn.collidepoint(pos):
                        self.human_first = not self.human_first
                    elif model_btn.collidepoint(pos):
                        path = self.open_file_dialog()
                        self.load_model(path)
                    elif start_btn.collidepoint(pos):
                        menu_running = False

    def draw_board(self):
        self.screen.fill(BLACK)
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

    def process_move(self, col, piece, name, color):
        if self.env.is_valid_location(col):
            row = self.env.get_next_open_row(col)
            self.env.drop_piece(row, col, piece)
            
            # 1. Draw the board first so the new piece appears
            self.draw_board()
            
            # 2. Print the winning text on top if the game ends
            if self.env.winning_move(piece):
                label = self.font.render(f"{name} wins!!", 1, color)
                self.screen.blit(label, (40, 10))
                pygame.display.update()
                self.env.game_over = True
                
            return True
        return False

    def play(self):
        self.draw_board()
        turn = 0 
        
        # Assign pieces based on who goes first
        human_piece = 1 if self.human_first else 2
        ai_piece = 2 if self.human_first else 1
        human_color = RED if human_piece == 1 else YELLOW
        ai_color = YELLOW if ai_piece == 2 else RED

        while not self.env.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                # HUMAN TURN
                is_human_turn = (turn == 0 and self.human_first) or (turn == 1 and not self.human_first)
                
                if event.type == pygame.MOUSEBUTTONDOWN and is_human_turn:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))
                    
                    if self.process_move(col, human_piece, "Human", human_color):
                        turn += 1
                        turn = turn % 2

            # AI TURN
            is_ai_turn = (turn == 0 and not self.human_first) or (turn == 1 and self.human_first)
            
            if is_ai_turn and not self.env.game_over:
                valid_moves = self.env.get_valid_locations()
                
                # Use loaded model, or play randomly if none loaded
                if self.agent:
                    col = self.agent.get_action(self.env.board.copy(), valid_moves, {})
                else:
                    col = random.choice(valid_moves)
                
                pygame.time.wait(500) 
                
                if self.process_move(col, ai_piece, "Agent", ai_color):
                    turn += 1
                    turn = turn % 2

        pygame.time.wait(3000)

if __name__ == "__main__":
    env = Connect4Env()
    ui = GameUI(env)
    ui.main_menu()
    ui.play()