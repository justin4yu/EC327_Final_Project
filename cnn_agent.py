import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class Connect4CNN(nn.Module):
    def __init__(self):
        super(Connect4CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class RLAgent:
    def __init__(self, epsilon=1.0, alpha=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Connect4CNN().to(self.device)
        self.target_model = Connect4CNN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

    def format_state(self, board, agent_piece=2):
        state = np.zeros((2, 6, 7), dtype=np.float32)
        opponent_piece = 1 if agent_piece == 2 else 2
        
        state[0][board == agent_piece] = 1
        state[1][board == opponent_piece] = 1
        return state

    def simulate_drop(self, board, row, col, piece):
        temp_board = board.copy()
        temp_board[row][col] = piece
        return temp_board

    def check_win(self, board, piece):
        for c in range(7-3):
            for r in range(6):
                if all(board[r][c+i] == piece for i in range(4)): return True
        for c in range(7):
            for r in range(6-3):
                if all(board[r+i][c] == piece for i in range(4)): return True
        for c in range(7-3):
            for r in range(6-3):
                if all(board[r+i][c+i] == piece for i in range(4)): return True
        for c in range(7-3):
            for r in range(3, 6):
                if all(board[r-i][c+i] == piece for i in range(4)): return True
        return False

    def get_setup_reward(self, board, piece):
        reward = 0
        opp_piece = 1 if piece == 2 else 2
        windows = []
        
        for r in range(6):
            for c in range(4): windows.append(list(board[r, c:c+4]))
        for c in range(7):
            for r in range(3): windows.append(list(board[r:r+4, c]))
        for r in range(3):
            for c in range(4): windows.append([board[r+i][c+i] for i in range(4)])
        for r in range(3, 6):
            for c in range(4): windows.append([board[r-i][c+i] for i in range(4)])
                
        for window in windows:
            if window.count(piece) == 3 and window.count(0) == 1: reward += 0.1  
            elif window.count(opp_piece) == 3 and window.count(0) == 1: reward -= 0.2  

        return reward

    def get_next_open_row(self, board, col):
        for r in range(6):
            if board[r][col] == 0: return r
        return -1

    def get_safe_locations(self, board, valid_locations, agent_piece):
        safe_locations = []
        opp_piece = 1 if agent_piece == 2 else 2
        
        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = self.simulate_drop(board, row, col, agent_piece)
            
            # Check if playing here lets the opponent win on top of it next turn
            if row < 5: 
                temp_board_2 = self.simulate_drop(temp_board, row + 1, col, opp_piece)
                if self.check_win(temp_board_2, opp_piece):
                    continue 
                    
            safe_locations.append(col)
            
        # Accept defeat if every move is a suicide move
        return safe_locations if safe_locations else valid_locations

    def heuristic_action(self, board, valid_locations, agent_piece):
        opponent_piece = 1 if agent_piece == 2 else 2
        
        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = self.simulate_drop(board, row, col, agent_piece)
            if self.check_win(temp_board, agent_piece): return col
                
        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = self.simulate_drop(board, row, col, opponent_piece)
            if self.check_win(temp_board, opponent_piece): return col
                
        return None

    def get_action(self, board, valid_locations, agent_piece=2):
        h_action = self.heuristic_action(board, valid_locations, agent_piece)
        if h_action is not None: return h_action

        safe_locations = self.get_safe_locations(board, valid_locations, agent_piece)

        if np.random.rand() <= self.epsilon:
            return random.choice(safe_locations)

        state = self.format_state(board, agent_piece)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()
            
        q_values[3] += 0.2
        q_values[2] += 0.1
        q_values[4] += 0.1

        for i in range(7):
            if i not in safe_locations:
                q_values[i] = -float('inf')
                
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size: return

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Mask out columns that are full in the next_state to prevent invalid Q-value learning
        next_q = self.target_model(next_states)
        valid_mask = (next_states[:, 0, 5, :] == 0) & (next_states[:, 1, 5, :] == 0)
        next_q = next_q.masked_fill(~valid_mask, -1e9)
        
        next_q_max = next_q.max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q_max

        loss = F.mse_loss(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay