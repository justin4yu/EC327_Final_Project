import random
import numpy as np
from connect4_engine import Connect4Env, ROWS, COLS
from q_agent import RLAgent

def evaluate_window(window, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= 90 

    return score

def calculate_custom_reward(board, piece):
    score = 0
    center_col = COLS // 2
    
    center_array = [int(i) for i in list(board[:, center_col])]
    score += center_array.count(piece) * 3

    left_inner = [int(i) for i in list(board[:, center_col - 1])]
    right_inner = [int(i) for i in list(board[:, center_col + 1])]
    score += (left_inner.count(piece) + right_inner.count(piece)) * 2

    left_outer = [int(i) for i in list(board[:, center_col - 2])]
    right_outer = [int(i) for i in list(board[:, center_col + 2])]
    score += (left_outer.count(piece) + right_outer.count(piece)) * 1

    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLS-3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    for c in range(COLS):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROWS-3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    return score

def train_headless():
    env = Connect4Env()
    agent = RLAgent(epsilon=0.3)
    
    try:
        agent.load("q_table.pkl")
        print("Loaded existing Q-table. Resuming training.")
    except FileNotFoundError:
        print("No existing Q-table found. Starting fresh.")

    episode = 0
    print("Starting continuous training. Press Ctrl+C to stop and save.")

    try:
        while True:
            env.reset()
            done = False
            
            while not done:
                valid_moves = env.get_valid_locations()
                if not valid_moves:
                    break
                    
                state_copy = env.board.copy()
                action = agent.get_action(state_copy, valid_moves, {})
                
                row = env.get_next_open_row(action)
                env.drop_piece(row, action, 2)
                
                if env.winning_move(2):
                    agent.train_step(state_copy, action, 100, env.board, [], True)
                    break
                    
                valid_moves_opp = env.get_valid_locations()
                if not valid_moves_opp:
                    agent.train_step(state_copy, action, 0, env.board, valid_moves_opp, True)
                    break
                    
                opp_action = random.choice(valid_moves_opp)
                opp_row = env.get_next_open_row(opp_action)
                env.drop_piece(opp_row, opp_action, 1)
                
                if env.winning_move(1):
                    agent.train_step(state_copy, action, -100, env.board, [], True)
                    break
                
                next_valid_moves = env.get_valid_locations()
                is_draw = len(next_valid_moves) == 0
                
                # The training script now calculates the reward
                position_reward = calculate_custom_reward(env.board, 2)
                step_penalty = -0.5
                total_reward = position_reward + step_penalty
                
                agent.train_step(state_copy, action, total_reward, env.board, next_valid_moves, is_draw)
                
                if is_draw:
                    break
            
            episode += 1
            if episode % 1000 == 0:
                print(f"Completed episode {episode}.")
                
    except KeyboardInterrupt:
        print(f"\nTraining interrupted manually. Total episodes this run: {episode}.")
        
    finally:
        agent.save()
        print("Training complete. Saved to q_table.pkl.")

if __name__ == "__main__":
    train_headless()