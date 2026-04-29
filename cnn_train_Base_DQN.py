import random
import numpy as np
import torch
import os
import shutil
from connect4_engine import Connect4Env, ROWS, COLS
from cnn_agent import RLAgent

def get_immediate_win_move(board, piece, valid_moves, env):
    for col in valid_moves:
        temp_board = board.copy()
        temp_row = env.get_next_open_row(col)
        temp_board[temp_row][col] = piece
        if env.winning_move(piece, temp_board):
            return col
    return None

def train_cnn():
    env = Connect4Env()
    agent = RLAgent(epsilon=0.9)

    print(f"Training on: {agent.device}")
    if agent.device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    try:
        if os.path.exists("cnn_model_v3.pth"):
            shutil.copy("cnn_model_v3.pth", "cnn_model_v3_backup.pth")
            print("Backed up existing model to cnn_model_v3_backup.pth")
        agent.load("cnn_model_v3.pth")
        print("Loaded existing model. Resuming training.")
    except FileNotFoundError:
        print("No existing model found. Starting fresh.")

    episode = 0
    recent_results = []
    RESULTS_WINDOW = 200
    EPSILON_MAX = 0.9
    EPSILON_MIN = 0.05
    self_play_enabled = False

    print("Starting training. Press Ctrl+C to stop and save.")

    try:
        while True:
            env.reset()
            done = False
            outcome = "draw"

            while not done:
                valid_moves = env.get_valid_locations()
                if not valid_moves:
                    break

                state_copy = env.board.copy()

                winning_move = get_immediate_win_move(env.board, 2, valid_moves, env)
                blocking_move = get_immediate_win_move(env.board, 1, valid_moves, env)

                if winning_move is not None:
                    action = winning_move
                elif blocking_move is not None:
                    action = blocking_move
                else:
                    action = agent.get_action(state_copy, valid_moves, {})

                row = env.get_next_open_row(action)
                blocked = (blocking_move is not None and action == blocking_move)

                env.drop_piece(row, action, 2)

                if env.winning_move(2):
                    agent.train_step(state_copy, action, 1.0, env.board, [], True, is_priority=True)
                    outcome = "win"
                    break

                valid_moves_opp = env.get_valid_locations()
                if not valid_moves_opp:
                    agent.train_step(state_copy, action, 0.0, env.board, [], True)
                    break

                opp_action = None

                opp_win = get_immediate_win_move(env.board, 1, valid_moves_opp, env)
                opp_block = get_immediate_win_move(env.board, 2, valid_moves_opp, env)

                if opp_win is not None:
                    opp_action = opp_win
                elif opp_block is not None:
                    opp_action = opp_block
                elif self_play_enabled and random.random() < 0.15:  # reduced from 0.3
                    opp_board = np.where(env.board == 1, 2, np.where(env.board == 2, 1, 0))
                    opp_action = agent.get_action(opp_board, valid_moves_opp, {})
                else:
                    opp_action = random.choice(valid_moves_opp)

                opp_row = env.get_next_open_row(opp_action)
                env.drop_piece(opp_row, opp_action, 1)

                if env.winning_move(1):
                    agent.train_step(state_copy, action, -1.0, env.board, [], True, is_priority=True)
                    outcome = "loss"
                    break

                next_valid_moves = env.get_valid_locations()
                is_draw = len(next_valid_moves) == 0

                center_col = COLS // 2
                center_array = [int(i) for i in list(env.board[:, center_col])]
                center_score = center_array.count(2) * 0.03
                blocking_bonus = 0.3 if blocked else 0.0
                total_reward = center_score + blocking_bonus
                is_priority = blocked

                agent.train_step(state_copy, action, total_reward, env.board, next_valid_moves, is_draw, is_priority=is_priority)

                if is_draw:
                    break

            recent_results.append(outcome)
            if len(recent_results) > RESULTS_WINDOW:
                recent_results.pop(0)

            if len(recent_results) == RESULTS_WINDOW:
                win_rate  = recent_results.count("win")  / RESULTS_WINDOW
                loss_rate = recent_results.count("loss") / RESULTS_WINDOW

                if loss_rate > 0.6:
                    agent.epsilon = min(agent.epsilon + 0.05, EPSILON_MAX)
                elif loss_rate > 0.4:
                    agent.epsilon = min(agent.epsilon + 0.01, EPSILON_MAX)
                elif win_rate > 0.6:
                    agent.epsilon = max(agent.epsilon - 0.01, EPSILON_MIN)
                    if not self_play_enabled and win_rate > 0.65:
                        self_play_enabled = True
                        print(f"*** Self-play unlocked at episode {episode} (win rate: {win_rate:.2f}) ***")

                # Lock self-play back out if win rate collapses
                if self_play_enabled and win_rate < 0.45:
                    self_play_enabled = False
                    agent.epsilon = 0.3
                    print(f"*** Self-play locked out at episode {episode} (win rate: {win_rate:.2f}) — epsilon reset to 0.3 ***")

            episode += 1
            if episode % 200 == 0:
                wins   = recent_results.count("win")
                losses = recent_results.count("loss")
                draws  = recent_results.count("draw")
                print(f"Episode {episode} | W{wins} L{losses} D{draws} / {len(recent_results)} | Epsilon: {agent.epsilon:.4f} | Self-play: {self_play_enabled}")

    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Total episodes this run: {episode}.")

    finally:
        agent.save("cnn_model_v3.pth")
        print("Training complete. Progress saved.")

if __name__ == "__main__":
    train_cnn()