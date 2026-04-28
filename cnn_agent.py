import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class Connect4CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Expanded filters and added a 3rd layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class RLAgent:
    def __init__(self, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The Active Network (learns every step)
        self.model = Connect4CNN().to(self.device)
        
        # The Target Network (frozen, calculates future rewards)
        self.target_model = Connect4CNN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() 
        
        # Lowered learning rate slightly for stability
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        
        self.memory = deque(maxlen=20000) # Doubled memory size
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps = 0 # Track steps to sync target network

    def get_action(self, state, valid_moves, info=None):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze().cpu().numpy()
        
        valid_q = [q_values[m] for m in valid_moves]
        max_q = max(valid_q)
        best_moves = [m for m in valid_moves if q_values[m] == max_q]
        return random.choice(best_moves)

    def train_step(self, state, action, reward, next_state, next_valid_moves, done):
        self.memory.append((state, action, reward, next_state, next_valid_moves, done))
        
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([b[0] for b in batch])).unsqueeze(1).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor([b[5] for b in batch]).to(self.device)

        # Active network calculates current value
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Frozen TARGET network calculates future value
        with torch.no_grad():
            next_q_all = self.target_model(next_states)
            next_q = torch.zeros(self.batch_size).to(self.device)
            for i in range(self.batch_size):
                valid_m = batch[i][4]
                if valid_m and not batch[i][5]:
                    next_q[i] = max([next_q_all[i][m] for m in valid_m])

        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Sync the networks every 1000 steps
        self.steps += 1
        if self.steps % 1000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename="cnn_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="cnn_model.pth"):
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()
        self.target_model.eval()