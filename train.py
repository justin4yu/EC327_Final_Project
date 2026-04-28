import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import the environment and the base agent
from connect4_engine import Connect4Env

# We'll define two agent classes: QLearningAgent and CNNAgent

def default_q_value():
    return np.zeros(7)

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.q_table = defaultdict(default_q_value)  # 7 actions (columns 0-6)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration

    def get_action(self, state, valid_moves, info=None):
        # Convert state to a tuple for hashing
        state_tuple = tuple(map(tuple, state))

        # Exploration vs exploitation
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            # Exploitation: choose the best valid action
            q_values = self.q_table[state_tuple]
            # Mask invalid moves with -infinity
            masked_q = np.full(7, -np.inf)
            for move in valid_moves:
                masked_q[move] = q_values[move]
            return np.argmax(masked_q)

    def train_step(self, state, action, reward, next_state, done):
        state_tuple = tuple(map(tuple, state))
        next_state_tuple = tuple(map(tuple, next_state))

        # Q-learning update
        current_q = self.q_table[state_tuple][action]
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_tuple])
            target_q = reward + self.gamma * next_max_q

        # Update Q-value
        self.q_table[state_tuple][action] += self.lr * (target_q - current_q)

        # Decay exploration rate
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class CNNAgent(nn.Module):
    def __init__(self):
        super(CNNAgent, self).__init__()
        # Input: 1 channel (the board), 6x7
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 6 * 7, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 actions

    def forward(self, x):
        # x shape: (batch, 1, 6, 7)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_qlearning_agent(episodes=1000):
    env = Connect4Env()
    agent = QLearningAgent()

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            valid_moves = env.get_valid_locations()
            if not valid_moves:
                break

            action = agent.get_action(state, valid_moves)
            row = env.get_next_open_row(action)
            env.drop_piece(row, action, 2)  # Agent is piece 2

            # Check if agent won
            if env.winning_move(2):
                reward = 1
                done = True
            else:
                # Check if human (piece 1) won on their turn? Actually, we need to simulate the opponent.
                # For simplicity, we'll let the opponent play randomly.
                # But note: the environment expects the opponent to be human in the UI.
                # For training, we'll have the agent play against a random opponent.

                # Opponent's turn (random)
                valid_moves_opponent = env.get_valid_locations()
                if valid_moves_opponent:
                    opp_action = random.choice(valid_moves_opponent)
                    opp_row = env.get_next_open_row(opp_action)
                    env.drop_piece(opp_row, opp_action, 1)  # Opponent is piece 1

                    if env.winning_move(1):
                        reward = -1  # Agent lost
                        done = True
                    else:
                        reward = 0  # Continue
                        # Check for draw
                        if len(env.get_valid_locations()) == 0:
                            done = True
                else:
                    # No valid moves for opponent -> draw
                    reward = 0
                    done = True

            # Agent learns from this step
            next_state = env.board.copy()
            agent.train_step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1}/{episodes}, Average Reward (last 100): {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")

    return agent, rewards_per_episode

def train_cnn_agent(episodes=1000):
    env = Connect4Env()
    agent = CNNAgent()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            valid_moves = env.get_valid_locations()
            if not valid_moves:
                break

            # Convert state to tensor: (1, 1, 6, 7)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            q_values_tensor = agent(state_tensor)  # shape: (1, 7)
            q_values = q_values_tensor.squeeze(0).detach().numpy()  # for epsilon-greedy

            # Epsilon-greedy (we'll use a fixed exploration for simplicity in this loop)
            # In practice, we'd decay epsilon over episodes.
            epsilon = max(0.01, 1.0 - episode/episodes)  # Linear decay
            if random.random() < epsilon:
                action = random.choice(valid_moves)
            else:
                # Mask invalid moves
                masked_q = np.full(7, -np.inf)
                for move in valid_moves:
                    masked_q[move] = q_values[move]
                action = np.argmax(masked_q)

            # Agent action
            row = env.get_next_open_row(action)
            env.drop_piece(row, action, 2)

            # Check if agent won
            if env.winning_move(2):
                reward = 1
                done = True
            else:
                # Opponent's turn (random)
                valid_moves_opponent = env.get_valid_locations()
                if valid_moves_opponent:
                    opp_action = random.choice(valid_moves_opponent)
                    opp_row = env.get_next_open_row(opp_action)
                    env.drop_piece(opp_row, opp_action, 1)

                    if env.winning_move(1):
                        reward = -1
                        done = True
                    else:
                        reward = 0
                        if len(env.get_valid_locations()) == 0:
                            done = True
                else:
                    reward = 0
                    done = True

            # Prepare for next step (state after opponent's move, which is agent's turn again)
            next_state = env.board.copy()
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
            next_q_values_tensor = agent(next_state_tensor)  # (1, 7)

            # Compute target Q-value
            if done:
                target_q = reward
            else:
                # Get max Q-value for next state over valid moves (for the agent)
                valid_moves_next = env.get_valid_locations()
                if valid_moves_next:
                    max_next_q = torch.max(next_q_values_tensor[0, valid_moves_next])
                    target_q = reward + 0.9 * max_next_q.item()
                else:
                    target_q = reward  # no valid moves, game over

            # Update the Q-value for the action taken
            # We want to set the target for the action we took to target_q, and keep the others the same as current prediction
            target = q_values_tensor.clone()  # (1, 7)
            target[0, action] = target_q

            # Optimize
            optimizer.zero_grad()
            loss = criterion(q_values_tensor, target)
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1}/{episodes}, Average Reward (last 100): {avg_reward:.3f}")

    return agent, rewards_per_episode

if __name__ == "__main__":
    print("Training Q-Learning Agent...")
    q_agent, q_rewards = train_qlearning_agent(episodes=500)

    print("\nTraining CNN Agent...")
    cnn_agent, cnn_rewards = train_cnn_agent(episodes=500)

    # Save the agents
    torch.save(q_agent.q_table, "q_agent_table.pth")
    torch.save(cnn_agent.state_dict(), "cnn_agent.pth")

    print("\nTraining complete. Agents saved.")