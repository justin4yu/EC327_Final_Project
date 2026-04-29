import random
import os
import numpy as np
from connect4_engine import Connect4Env
from cnn_agent import SupervisedAgent
from bitbully_wrapper import get_bitbully_score

def generate_game_data(env, cache):
    env.reset()
    states_and_scores = []
    move_history = []
    
    while True:
        # --- PLAYER 1 (Opponent) MOVES ---
        valid_moves = env.get_valid_locations()
        if not valid_moves: break
        
        action = random.choice(valid_moves)
        row = env.get_next_open_row(action)
        env.drop_piece(row, action, 1)
        move_history.append(action)

        # Check if Player 1 won. If so, P2 (Agent) lost. 
        # We don't call BitBully because the state is terminal.
        if env.winning_move(1): break

        # --- DATA COLLECTION POINT (Player 2's Turn) ---
        # Check if board became full after Player 1's move
        valid_moves = env.get_valid_locations()
        if not valid_moves: break

        state_copy = env.board.copy()
        board_bytes = state_copy.tobytes()
        
        if board_bytes in cache:
            score = cache[board_bytes]
        else:
            # We call bitbully here. Since it is P2's turn, 
            # it returns the value for P2.
            score = get_bitbully_score(move_history)
            cache[board_bytes] = score
            
        states_and_scores.append((state_copy, score))

        # --- PLAYER 2 (Your Agent) MOVES ---
        action2 = random.choice(valid_moves)
        row2 = env.get_next_open_row(action2)
        env.drop_piece(row2, action2, 2)
        move_history.append(action2)
        
        # Check if P2 (Agent) won.
        if env.winning_move(2): break
            
    return states_and_scores

def train():
    env = Connect4Env()
    agent = SupervisedAgent()
    cache = {}
    save_path = "cnn_supervised.pth"
    
    # Import torch here for the print check
    import torch 
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Training on: {agent.device}")
    
    if os.path.exists(save_path):
        agent.load(save_path)
        print("Loaded existing model checkpoint.")
        
    print("Starting infinite supervised training. Press Ctrl+C (or stop in Colab) to save and exit.")
    
    episode = 1
    try:
        while True:  # This makes it run infinitely
            game_data = generate_game_data(env, cache)
            
            for state, score in game_data:
                agent.collect_experience(state, score)
                
            loss = agent.train_step()
            
            if episode % 100 == 0:
                print(f"Episode {episode} | Cache: {len(cache)} | Loss: {loss:.4f}")
                
            if episode % 1000 == 0:
                agent.save(save_path)
                print(f"Checkpoint saved to {save_path}")
                
            episode += 1  # Increment the counter manually
                
    except KeyboardInterrupt:
        print("\nManual stop detected.")
    finally:
        agent.save(save_path)
        print("Final model saved successfully.")

if __name__ == "__main__":
    train()