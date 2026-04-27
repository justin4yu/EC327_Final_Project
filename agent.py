# agent.py
"""
DQN Agent with a small CNN for Connect 4.

ARCHITECTURE
------------
Input:  3 x 6 x 7 tensor — one binary channel each for:
            channel 0: agent's pieces   (1 where agent placed, 0 elsewhere)
            channel 1: opponent's pieces
            channel 2: empty squares
Output: 7 Q-values, one per column

The CNN learns spatial patterns (3-in-a-row, blocked threats, forks) that
tabular Q-learning can never generalise across.

EXPERIENCE REPLAY
-----------------
Instead of learning from each transition immediately (which causes
correlated, unstable updates), we store transitions in a ReplayBuffer
and sample random mini-batches. This breaks the correlation and stabilises
training — the key innovation of the original DQN paper.

TARGET NETWORK
--------------
A second "frozen" copy of the network is used to compute TD targets.
It is synced with the main network every TARGET_UPDATE episodes.
Without this the target keeps shifting every step, making training diverge.

USAGE
-----
Training:   python train.py          (saves dqn_weights.pth)
Playing:    python connect4_engine.py (loads dqn_weights.pth, epsilon=0)
"""

import random
import math
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ROWS = 6
COLS = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Neural Network ─────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Small CNN that maps a board state to Q-values for each column.

    Conv layers detect local patterns (threats, open lines).
    FC layers combine them into a global position evaluation.
    """
    def __init__(self):
        super().__init__()
        # 3 input channels (agent / opponent / empty), each 6x7
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # 64 x 6 x 7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 x 6 x 7
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # 128 x 6 x 7

        flat_size = 128 * ROWS * COLS   # 128 * 42 = 5376

        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, COLS)   # one Q-value per column

    def forward(self, x):
        # x: (batch, 3, 6, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)        # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)               # raw Q-values, no activation


# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer that stores (state, action, reward, next_state, done)
    transitions and returns random mini-batches for training.
    """
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class RLAgent:
    """
    DQN agent. Drop-in replacement for the old Q-table agent —
    connect4_engine.py and train.py call the same get_action / train_step /
    end_episode interface.
    """

    def __init__(
        self,
        lr:             float = 1e-3,
        gamma:          float = 0.99,
        epsilon:        float = 1.0,
        epsilon_min:    float = 0.05,
        epsilon_decay:  float = 0.9995,
        batch_size:     int   = 64,
        target_update:  int   = 200,      # sync target net every N episodes
        weights_path:   str   = "dqn_weights.pth",
        # legacy kwarg ignored so old code doesn't crash
        q_table_path:   str   = "q_table.pkl",
    ):
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.weights_path  = weights_path

        # Main network (trained every step) + frozen target network
        self.policy_net = DQNNetwork().to(DEVICE)
        self.target_net = DQNNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer()

        # Bookkeeping
        self.episode_count = 0
        self.wins   = 0
        self.losses = 0
        self.draws  = 0
        self._loss_log = []

        self._load_weights()
        print(f"[DQN Agent] Device={DEVICE}  epsilon={self.epsilon:.3f}")

    # ------------------------------------------------------------------
    # Public interface (unchanged from Q-table agent)
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray, valid_moves: list, info: dict) -> int:
        """Epsilon-greedy action selection using the CNN."""
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        tensor = self._board_to_tensor(state).unsqueeze(0).to(DEVICE)  # (1,3,6,7)
        with torch.no_grad():
            q_values = self.policy_net(tensor).squeeze(0).cpu().numpy()  # (7,)

        # Mask invalid columns to -inf so argmax never picks them
        masked = np.full(COLS, -np.inf)
        for col in valid_moves:
            masked[col] = q_values[col]

        return int(np.argmax(masked))

    def train_step(
        self,
        state:            np.ndarray,
        action:           int,
        reward:           float,
        next_state:       np.ndarray,
        done:             bool,
        next_valid_moves: list = None,
    ) -> float:
        """Store transition and, once the buffer is big enough, do one gradient update."""
        self.replay.push(
            self._board_to_tensor(state).numpy(),
            action,
            reward,
            self._board_to_tensor(next_state).numpy(),
            float(done),
        )

        if len(self.replay) < self.batch_size:
            return 0.0

        return self._update()

    def end_episode(self, outcome: str = "draw"):
        """Decay epsilon, sync target net periodically, log progress."""
        self.episode_count += 1
        if outcome == "win":    self.wins   += 1
        elif outcome == "loss": self.losses += 1
        else:                   self.draws  += 1

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.episode_count % 500 == 0:
            avg_loss = np.mean(self._loss_log[-500:]) if self._loss_log else 0
            self._save_weights()
            print(
                f"[DQN] ep={self.episode_count:>6}  "
                f"W/L/D={self.wins}/{self.losses}/{self.draws}  "
                f"eps={self.epsilon:.4f}  "
                f"buf={len(self.replay)}  "
                f"loss={avg_loss:.4f}"
            )

    # legacy alias so train.py that calls save_q_table() still works
    def save_q_table(self):
        self._save_weights()

    # ------------------------------------------------------------------
    # Internal: gradient update
    # ------------------------------------------------------------------

    def _update(self) -> float:
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t      = torch.tensor(states,      device=DEVICE)          # (B,3,6,7)
        next_states_t = torch.tensor(next_states, device=DEVICE)
        actions_t     = torch.tensor(actions,     device=DEVICE)          # (B,)
        rewards_t     = torch.tensor(rewards,     device=DEVICE)          # (B,)
        dones_t       = torch.tensor(dones,       device=DEVICE)          # (B,)

        # Q(s, a) from the policy network
        q_pred = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a')  — computed with no gradient
        with torch.no_grad():
            q_next  = self.target_net(next_states_t).max(dim=1).values
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_pred, q_target)   # Huber loss — more stable than MSE

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients early in training
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self._loss_log.append(loss_val)
        return loss_val

    # ------------------------------------------------------------------
    # Internal: board encoding
    # ------------------------------------------------------------------

    def _board_to_tensor(self, board: np.ndarray) -> torch.Tensor:
        """
        Encode the 6x7 board as a 3-channel binary tensor:
            ch0: 1 where agent (piece=2) has a piece
            ch1: 1 where opponent (piece=1) has a piece
            ch2: 1 where the cell is empty
        This gives the CNN three clean, independent feature maps.
        """
        ch_agent   = (board == 2).astype(np.float32)
        ch_opponent= (board == 1).astype(np.float32)
        ch_empty   = (board == 0).astype(np.float32)
        tensor = np.stack([ch_agent, ch_opponent, ch_empty], axis=0)  # (3,6,7)
        return torch.tensor(tensor)

    # ------------------------------------------------------------------
    # Internal: save / load
    # ------------------------------------------------------------------

    def _save_weights(self):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "episodes": self.episode_count,
        }, self.weights_path)

    def _load_weights(self):
        if not os.path.exists(self.weights_path):
            print("[DQN Agent] No saved weights — starting fresh.")
            return
        try:
            ckpt = torch.load(self.weights_path, map_location=DEVICE)
            self.policy_net.load_state_dict(ckpt["policy"])
            self.target_net.load_state_dict(ckpt["target"])
            self.epsilon       = ckpt.get("epsilon",  self.epsilon_min)
            self.episode_count = ckpt.get("episodes", 0)
            print(f"[DQN Agent] Loaded weights from '{self.weights_path}'  "
                  f"ep={self.episode_count}  eps={self.epsilon:.4f}")
        except Exception as e:
            print(f"[DQN Agent] Could not load weights ({e}) — starting fresh.")