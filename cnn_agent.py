import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

COLS = 7

def get_split_state(board):
    agent_pieces = (board == 2).astype(np.float32)
    opp_pieces = (board == 1).astype(np.float32)
    return np.stack([agent_pieces, opp_pieces])

class Connect4CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
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

        self.model = Connect4CNN().to(self.device)
        self.target_model = Connect4CNN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

        self.memory = deque(maxlen=80000)
        self.priority_memory = deque(maxlen=20000)
        self.batch_size = 512
        self.train_freq = 4

        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps = 0

    def _make_mask(self, valid_moves):
        mask = np.zeros(COLS, dtype=bool)
        for m in valid_moves:
            mask[m] = True
        return mask

    def get_action(self, state, valid_moves, info=None):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        split_s = get_split_state(state)
        state_tensor = torch.FloatTensor(split_s).unsqueeze(0).to(self.device, non_blocking=True)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze().cpu().numpy()

        valid_q = [q_values[m] for m in valid_moves]
        max_q = max(valid_q)
        best_moves = [m for m in valid_moves if q_values[m] == max_q]
        return random.choice(best_moves)

    def train_step(self, state, action, reward, next_state, next_valid_moves, done, is_priority=False):
        next_mask = self._make_mask(next_valid_moves)
        split_s = get_split_state(state)
        split_next_s = get_split_state(next_state)
        
        experience = (split_s, action, reward, split_next_s, next_mask, done)
        
        if is_priority:
            self.priority_memory.append(experience)
        else:
            self.memory.append(experience)

        self.steps += 1

        total_memory = len(self.memory) + len(self.priority_memory)
        if total_memory < self.batch_size:
            return
        if self.steps % self.train_freq != 0:
            return

        batch = []
        priority_count = min(len(self.priority_memory), self.batch_size // 4)
        regular_count = self.batch_size - priority_count
        
        if priority_count > 0:
            batch.extend(random.sample(self.priority_memory, priority_count))
        
        if len(self.memory) >= regular_count:
            batch.extend(random.sample(self.memory, regular_count))
        else:
            return 

        states = torch.FloatTensor(
            np.array([b[0] for b in batch])
        ).to(self.device, non_blocking=True)

        actions = torch.LongTensor(
            [b[1] for b in batch]
        ).unsqueeze(1).to(self.device, non_blocking=True)

        rewards = torch.FloatTensor(
            [b[2] for b in batch]
        ).to(self.device, non_blocking=True)

        next_states = torch.FloatTensor(
            np.array([b[3] for b in batch])
        ).to(self.device, non_blocking=True)

        valid_masks = torch.BoolTensor(
            np.array([b[4] for b in batch])
        ).to(self.device, non_blocking=True)

        dones = torch.FloatTensor(
            [b[5] for b in batch]
        ).to(self.device, non_blocking=True)

        current_q = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_model(next_states)
            next_q_all[~valid_masks] = -float("inf")
            next_q = next_q_all.max(dim=1).values
            next_q[dones.bool()] = 0.0

        target_q = rewards + self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.steps % (1000 * self.train_freq) == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename="cnn_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="cnn_model.pth"):
        self.model.load_state_dict(
            torch.load(filename, map_location=self.device, weights_only=True)
        )
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.train()
        self.target_model.eval()