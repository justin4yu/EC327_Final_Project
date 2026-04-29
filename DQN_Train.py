import random
import numpy as np
import torch
from connect4_engine import Connect4Env, ROWS, COLS
from cnn_agent import RLAgent

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

def evaluate_wide_traps(board, piece):
    score = 0
    p = str(piece)
    
    trap_2_2 = p + p + "0" + p + p
    trap_2_3_a = p + p + "0" + p + p + p
    trap_2_3_b = p + p + p + "0" + p + p
    
    lines = []
    
    for r in range(ROWS):
        lines.append(board[r, :])
        
    for offset in range(-ROWS + 1, COLS):
        diag1 = np.diagonal(board, offset=offset)
        if len(diag1) >= 5: 
            lines.append(diag1)
            
        diag2 = np.diagonal(np.fliplr(board), offset=offset)
        if len(diag2) >= 5: 
            lines.append(diag2)
            
    for line in lines:
        line_str = "".join(map(lambda x: str(int(x)), line))
        if trap_2_3_a in line_str or trap_2_3_b in line_str:
            score += 30
        elif trap_2_2 in line_str:
            score += 15
            
    return score

def calculate_custom_reward(board, piece):
    score = 0
    
    score += evaluate_wide_traps(board, piece)
    
    center_col = COLS // 2

    center_array = [int(i) for i in list(board[:, center_col])]
    score += center_array.count(piece) * 3

    left_inner = [int(i) for i in list(board[:, center_col - 1])]
    right_inner = [int(i) for i in list(board[:, center_col + 1])]
    score += (left_inner.count(piece) + right_inner.count(piece)) * 2

    left_outer = [int(i) for i in list(board[:, center_col - 2])]
    right_outer = [int(i) for i in list(board[:, center_col + 2])]
    score += (left_outer.count(piece) + right_outer.count(piece)) * 1

    lines = []
    
    for r in range(ROWS): 
        lines.append(board[r, :])
    for c in range(COLS): 
        lines.append(board[:, c])
        
    for offset in range(-ROWS + 1, COLS):
        diag1 = np.diagonal(board, offset=offset)
        if len(diag1) >= 4: 
            lines.append(diag1)
            
        diag2 = np.diagonal(np.fliplr(board), offset=offset)
        if len(diag2) >= 4: 
            lines.append(diag2)
            
    for line in lines:
        line_list = [int(i) for i in list(line)]
        for i in range(len(line_list) - 3):
            window = line_list[i:i+4]
            score += evaluate_window(window, piece)

    return score

def train_cnn():
    env = Connect4Env()
    agent = RLAgent(epsilon=0.3)

    print(f"Training on: {agent.device}") 
    if agent.device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    try:
        agent.load("cnn_model_v2.pth")
        print("Loaded existing CNN model. Resuming training.")
    except FileNotFoundError:
        print("No existing CNN model found. Starting fresh.")

    episode = 0
    print("Starting CNN training. Press Ctrl+C to stop and save.")

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
                
                temp_board = env.board.copy()
                temp_board[row][action] = 1
                blocked = env.winning_move(1, temp_board)
                
                env.drop_piece(row, action, 2)

                if env.winning_move(2):
                    agent.train_step(state_copy, action, 100, env.board, [], True, is_priority=True)
                    break

                valid_moves_opp = env.get_valid_locations()
                if not valid_moves_opp:
                    agent.train_step(state_copy, action, 0, env.board, valid_moves_opp, True)
                    break

                opp_action = None
                
                for col in valid_moves_opp:
                    temp_board = env.board.copy()
                    temp_row = env.get_next_open_row(col)
                    temp_board[temp_row][col] = 1
                    if env.winning_move(1, temp_board):
                        opp_action = col
                        break

                if opp_action is None:
                    for col in valid_moves_opp:
                        temp_board = env.board.copy()
                        temp_row = env.get_next_open_row(col)
                        temp_board[temp_row][col] = 2 
                        if env.winning_move(2, temp_board):
                            opp_action = col
                            break

                if opp_action is None:
                    if random.random() < 0.5:
                        opp_board = np.where(env.board == 1, 2, np.where(env.board == 2, 1, 0))
                        opp_action = agent.get_action(opp_board, valid_moves_opp, {})
                    else:
                        opp_action = random.choice(valid_moves_opp)

                opp_row = env.get_next_open_row(opp_action)
                env.drop_piece(opp_row, opp_action, 1)

                if env.winning_move(1):
                    agent.train_step(state_copy, action, -100, env.board, [], True, is_priority=True)
                    break

                next_valid_moves = env.get_valid_locations()
                is_draw = len(next_valid_moves) == 0

                position_reward = calculate_custom_reward(env.board, 2)
                blocking_bonus = 30 if blocked else 0
                
                total_reward = position_reward + blocking_bonus
                is_priority = blocked

                agent.train_step(state_copy, action, total_reward, env.board, next_valid_moves, is_draw, is_priority=is_priority)

                if is_draw:
                    break

            episode += 1
            if episode % 100 == 0:
                print(f"Completed episode {episode}.")

    except KeyboardInterrupt:
        print(f"\nTraining interrupted manually. Total episodes this run: {episode}.")

    finally:
        agent.save("cnn_model_v2.pth")
        print("Training complete. Progress saved.")

if __name__ == "__main__":
    train_cnn()