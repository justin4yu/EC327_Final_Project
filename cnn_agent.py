import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

COLS = 7
ROWS = 6

def get_split_state(board):
    agent_pieces = (board == 2).astype(np.float32)
    opp_pieces = (board == 1).astype(np.float32)
    return np.stack([agent_pieces, opp_pieces])

# ---------------------------------------------------------
# OLD RL ARCHITECTURE (Preserved for your old .pth files)
# ---------------------------------------------------------
class Connect4CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 7) # 7 Outputs

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------------------------------------------
# NEW SUPERVISED ARCHITECTURE (1 Output)
# ---------------------------------------------------------
class SupervisedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 1) # 1 Output

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SupervisedAgent:
    def __init__(self, epsilon=0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SupervisedCNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.memory = []
        self.batch_size = 512

    def get_action(self, board, valid_moves, info=None):
        # Identical interface to RLAgent, so the engine doesn't need to change.
        best_score = -float('inf')
        best_col = random.choice(valid_moves)

        for col in valid_moves:
            temp_board = board.copy()
            row = -1
            for r in range(ROWS):
                if temp_board[r][col] == 0:
                    row = r
                    break

            if row != -1:
                temp_board[row][col] = 2

                # Immediate win check to grab a victory instantly
                if self._check_win(2, temp_board):
                    return col

                split_s = get_split_state(temp_board)
                state_tensor = torch.FloatTensor(split_s).unsqueeze(0).to(self.device, non_blocking=True)

                with torch.no_grad():
                    score = self.model(state_tensor).item()

                if score > best_score:
                    best_score = score
                    best_col = col

        return best_col

    def _check_win(self, piece, board):
        for c in range(COLS-3):
            for r in range(ROWS):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece: return True
        for c in range(COLS):
            for r in range(ROWS-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece: return True
        for c in range(COLS-3):
            for r in range(ROWS-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece: return True
        for c in range(COLS-3):
            for r in range(3, ROWS):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece: return True
        return False

    def collect_experience(self, state, true_score):
        split_s = get_split_state(state)
        self.memory.append((split_s, true_score))
        if len(self.memory) > 100000:
            self.memory.pop(0)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device, non_blocking=True)
        target_scores = torch.FloatTensor([b[1] for b in batch]).unsqueeze(1).to(self.device, non_blocking=True)
        predicted_scores = self.model(states)
        loss = nn.MSELoss()(predicted_scores, target_scores)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    def save(self, filename="cnn_supervised.pth"):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="cnn_supervised.pth"):
        self.model.load_state_dict(
            torch.load(filename, map_location=self.device, weights_only=True)
        )
        self.model.eval()

# ---------------------------------------------------------
# RLAgent — alias so connect4_engine.py can import it
# ---------------------------------------------------------
RLAgent = SupervisedAgent