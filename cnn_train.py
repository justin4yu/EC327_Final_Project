import sys
import torch
from connect4_engine import Connect4Env
from cnn_agent import RLAgent

def train():
    env = Connect4Env()
    agent = RLAgent(epsilon=1.0)
    
    print(f"Training on: {agent.device}") 
    if agent.device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    try:
        agent.load("cnn_model_v3.pth")
        print("Loaded existing CNN model. Resuming training.")
    except FileNotFoundError:
        print("No existing CNN model found. Starting fresh.")

    batch_size = 100
    print("Starting infinite self play training. Press Ctrl+C to stop and save.")

    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_moves = 0  # Tracks moves for the batch
    e = 1

    try:
        while True:
            env.reset()
            board = env.board.copy()
            done = False
            turn = 0
            moves = 0  # Tracks moves for this specific game

            # Store the last states and actions to assign delayed rewards
            last_state_p1 = None
            last_action_p1 = None
            last_state_p2 = None
            last_action_p2 = None

            while not done:
                valid_locations = env.get_valid_locations()
                current_piece = 1 if turn == 0 else 2
                
                # The same agent predicts moves for both players
                action = agent.get_action(board, valid_locations, agent_piece=current_piece)
                row = env.get_next_open_row(action)
                
                state_formatted = agent.format_state(board, current_piece)
                
                env.drop_piece(row, action, current_piece)
                moves += 1  # Increment the move counter
                
                next_board = env.board.copy()
                next_state_formatted = agent.format_state(next_board, current_piece)
                
                if env.winning_move(current_piece):
                    done = True
                    # Reward the winner
                    agent.remember(state_formatted, action, 1.0, next_state_formatted, True)
                    
                    # Punish the loser for their previous move
                    if current_piece == 1:
                        p1_wins += 1
                        if last_state_p2 is not None:
                            losing_state_p2 = agent.format_state(next_board, 2)
                            agent.remember(last_state_p2, last_action_p2, -1.0, losing_state_p2, True)
                    else:
                        p2_wins += 1
                        if last_state_p1 is not None:
                            losing_state_p1 = agent.format_state(next_board, 1)
                            agent.remember(last_state_p1, last_action_p1, -1.0, losing_state_p1, True)
                            
                elif len(env.get_valid_locations()) == 0:
                    done = True
                    draws += 1
                    # Give a partial reward to both players for a draw
                    agent.remember(state_formatted, action, 0.5, next_state_formatted, True)
                    
                    if current_piece == 1 and last_state_p2 is not None:
                        draw_state_p2 = agent.format_state(next_board, 2)
                        agent.remember(last_state_p2, last_action_p2, 0.5, draw_state_p2, True)
                    elif current_piece == 2 and last_state_p1 is not None:
                        draw_state_p1 = agent.format_state(next_board, 1)
                        agent.remember(last_state_p1, last_action_p1, 0.5, draw_state_p1, True)
                else:
                    reward = agent.get_setup_reward(next_board, current_piece)
                    agent.remember(state_formatted, action, reward, next_state_formatted, False)

                if current_piece == 1:
                    last_state_p1 = state_formatted
                    last_action_p1 = action
                else:
                    last_state_p2 = state_formatted
                    last_action_p2 = action
                
                board = next_board
                turn = 1 if turn == 0 else 0
                
                agent.replay()
                
            agent.decay_epsilon()
            total_moves += moves
            
            if e % batch_size == 0:
                agent.update_target_network()
                avg_moves = total_moves / batch_size
                print(f"Episode {e} | P1: {p1_wins} | P2: {p2_wins} | Draws: {draws} | Avg Moves: {avg_moves:.1f} | Eps: {agent.epsilon:.3f}")
                p1_wins = 0
                p2_wins = 0
                draws = 0
                total_moves = 0

            if e % 1000 == 0:
                agent.save(f"cnn_model_v3_checkpoint_{e}.pth")
                print(f"Checkpoint saved at episode {e}.")
            
            e += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress.")
        
    finally:
        agent.save("cnn_model_v3_final.pth")
        print("Final model saved.")

if __name__ == "__main__":
    train()