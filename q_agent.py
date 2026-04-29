import random
import numpy as np
import pickle

class RLAgent:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        return str(state.astype(int).flatten())

    def get_action(self, state, valid_moves, info):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)

        state_key = self.get_state_key(state)
        q_values = [self.q_table.get((state_key, move), 0.0) for move in valid_moves]
        max_q = max(q_values)
        
        best_moves = [m for m, q in zip(valid_moves, q_values) if q == max_q]
        return random.choice(best_moves)

    def train_step(self, state, action, reward, next_state, next_valid_moves, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        old_value = self.q_table.get((state_key, action), 0.0)
        
        if done:
            next_max = 0.0
        else:
            next_max = max([self.q_table.get((next_state_key, m), 0.0) for m in next_valid_moves], default=0.0)

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state_key, action)] = new_value

    def save(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)