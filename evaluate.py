import numpy as np
import random
import torch
import torch.nn as nn
from collections import defaultdict
from connect4_engine import Connect4Env

# Import the agent classes from train.py
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

    def get_action(self, state, valid_moves):
        # Convert state to tensor: (1, 1, 6, 7)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        q_values = self.forward(state_tensor)
        q_values = q_values.squeeze(0).detach().numpy()  # for epsilon-greedy

        # For evaluation, we'll use greedy (no exploration)
        # Mask invalid moves
        masked_q = np.full(7, -np.inf)
        for move in valid_moves:
            masked_q[move] = q_values[move]
        return np.argmax(masked_q)

def play_game(agent1, agent2, verbose=False):
    """Play a game between two agents. Returns 1 if agent1 wins, -1 if agent2 wins, 0 for draw."""
    env = Connect4Env()
    state = env.reset()
    done = False

    # Agent 1 is piece 1, Agent 2 is piece 2
    current_agent = agent1  # Start with agent 1
    current_piece = 1

    while not done:
        valid_moves = env.get_valid_locations()
        if not valid_moves:
            if verbose:
                print("No valid moves - draw")
            return 0  # Draw

        # Get action from current agent
        if isinstance(current_agent, CNNAgent):
            action = current_agent.get_action(state, valid_moves)
        else:  # QLearningAgent
            action = current_agent.get_action(state, valid_moves)

        # Execute action
        row = env.get_next_open_row(action)
        env.drop_piece(row, action, current_piece)

        # Check for win
        if env.winning_move(current_piece):
            if verbose:
                print(f"Piece {current_piece} wins!")
            return 1 if current_piece == 1 else -1

        # Switch players
        current_piece = 3 - current_piece  # Switch between 1 and 2
        current_agent = agent2 if current_agent == agent1 else agent1

        # Check for draw (board full)
        if len(env.get_valid_locations()) == 0:
            if verbose:
                print("Board full - draw")
            return 0

    return 0  # Shouldn't reach here

def evaluate_agent(agent, opponent_type="random", num_games=100):
    """Evaluate an agent against a specified opponent."""
    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        if opponent_type == "random":
            opponent = RandomAgent()
        elif opponent_type == "QLearning":
            # Load the Q-learning agent
            q_agent = QLearningAgent()
            q_agent.q_table = torch.load("q_agent_table.pth", weights_only=False)
            opponent = q_agent
        elif opponent_type == "CNN":
            # Load the CNN agent
            cnn_agent = CNNAgent()
            cnn_agent.load_state_dict(torch.load("cnn_agent.pth"))
            opponent = cnn_agent
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        # Play game (agent as player 1, opponent as player 2)
        result = play_game(agent, opponent)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

        # Also play with agent as player 2 to get fair evaluation
        result = play_game(opponent, agent)
        if result == -1:  # Opponent wins means agent loses
            wins += 1
        elif result == 1:  # Agent wins
            losses += 1
        else:
            draws += 1

    total_games = num_games * 2
    win_rate = wins / total_games if total_games > 0 else 0
    loss_rate = losses / total_games if total_games > 0 else 0
    draw_rate = draws / total_games if total_games > 0 else 0

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate
    }

class RandomAgent:
    def get_action(self, state, valid_moves):
        return random.choice(valid_moves)

if __name__ == "__main__":
    print("Loading trained agents...")

    # Load Q-Learning agent
    q_agent = QLearningAgent()
    q_agent.q_table = torch.load("q_agent_table.pth", weights_only=False)
    print("Q-Learning agent loaded.")

    # Load CNN agent
    cnn_agent = CNNAgent()
    cnn_agent.load_state_dict(torch.load("cnn_agent.pth"))
    print("CNN agent loaded.")

    print("\nEvaluating agents against random player (100 games each)...")

    # Evaluate Q-Learning agent
    q_results = evaluate_agent(q_agent, opponent_type="random", num_games=50)
    print(f"Q-Learning Agent vs Random: {q_results['wins']}W-{q_results['losses']}L-{q_results['draws']}D (Win Rate: {q_results['win_rate']:.2%})")

    # Evaluate CNN agent
    cnn_results = evaluate_agent(cnn_agent, opponent_type="random", num_games=50)
    print(f"CNN Agent vs Random: {cnn_results['wins']}W-{cnn_results['losses']}L-{cnn_results['draws']}D (Win Rate: {cnn_results['win_rate']:.2%})")

    print("\nEvaluating agents against each other (50 games each)...")

    # Evaluate Q-Learning vs CNN
    q_vs_cnn = evaluate_agent(q_agent, opponent_type="CNN", num_games=25)
    print(f"Q-Learning vs CNN: {q_vs_cnn['wins']}W-{q_vs_cnn['losses']}L-{q_vs_cnn['draws']}D (Win Rate: {q_vs_cnn['win_rate']:.2%})")

    # Evaluate CNN vs Q-Learning (already covered in the function, but let's be explicit)
    cnn_vs_q = evaluate_agent(cnn_agent, opponent_type="QLearning", num_games=25)
    print(f"CNN vs Q-Learning: {cnn_vs_q['wins']}W-{cnn_vs_q['losses']}L-{cnn_vs_q['draws']}D (Win Rate: {cnn_vs_q['win_rate']:.2%})")

    print("\nSummary:")
    print(f"Q-Learning Agent win rate vs Random: {q_results['win_rate']:.2%}")
    print(f"CNN Agent win rate vs Random: {cnn_results['win_rate']:.2%}")
    print(f"Q-Learning Agent win rate vs CNN: {q_vs_cnn['win_rate']:.2%}")
    print(f"CNN Agent win rate vs Q-Learning: {cnn_vs_q['win_rate']:.2%}")