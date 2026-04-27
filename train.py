# train.py
"""
Headless self-play training loop for the DQN Connect 4 agent.

The DQN agent (piece 2) trains against a Minimax opponent (piece 1) at
depth=1. This is much better than pure self-play early on because:
  - A random opponent teaches nothing about real threats
  - A strong Minimax (depth 5+) crushes the untrained agent every game,
    giving only loss signals and slow learning
  - Minimax depth=1 plays sensibly (wins/blocks if it can in one move)
    but is beatable, giving the DQN a mix of wins, losses and draws to
    learn from

Once the agent is trained you can increase OPPONENT_DEPTH to keep it
challenged, or switch to self-play by setting OPPONENT = "self".

Run:
    pip install torch numpy pygame
    python train.py

Then play:
    python connect4_engine.py
"""

import math
import random
import numpy as np
from connect4_engine import Connect4Env
from agent import RLAgent

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_EPISODES   = 300_000  # run to 300k total for a strong agent
OPPONENT_DEPTH = 2        # depth 2 now that agent has baseline skill at 78k

# ── Rewards ────────────────────────────────────────────────────────────────────
R_WIN   =  1.0
R_LOSS  = -1.0
R_DRAW  =  0.3   # small positive: a draw vs minimax is respectable


# ── Minimax opponent (no learning, just a fixed policy) ────────────────────────

def minimax(board, depth, alpha, beta, maximizing, piece):
    """Lightweight minimax used as the training opponent."""
    valid = [c for c in range(7) if board[5][c] == 0]

    def winning(b, p):
        for r in range(6):
            for c in range(4):
                if all(b[r][c+i] == p for i in range(4)): return True
        for c in range(7):
            for r in range(3):
                if all(b[r+i][c] == p for i in range(4)): return True
        for r in range(3):
            for c in range(4):
                if all(b[r+i][c+i] == p for i in range(4)): return True
        for r in range(3, 6):
            for c in range(4):
                if all(b[r-i][c+i] == p for i in range(4)): return True
        return False

    opp = 2 if piece == 1 else 1
    if winning(board, piece): return  1_000_000
    if winning(board, opp):   return -1_000_000
    if not valid or depth == 0: return 0

    if maximizing:
        val = -math.inf
        for col in valid:
            row = next(r for r in range(6) if board[r][col] == 0)
            temp = board.copy(); temp[row][col] = piece
            val = max(val, minimax(temp, depth-1, alpha, beta, False, piece))
            alpha = max(alpha, val)
            if alpha >= beta: break
        return val
    else:
        val = math.inf
        for col in valid:
            row = next(r for r in range(6) if board[r][col] == 0)
            temp = board.copy(); temp[row][col] = opp
            val = min(val, minimax(temp, depth-1, alpha, beta, True, piece))
            beta = min(beta, val)
            if alpha >= beta: break
        return val


def minimax_action(board, depth, piece):
    valid = [c for c in range(7) if board[5][c] == 0]
    best_col, best_score = valid[0], -math.inf
    for col in valid:
        row = next(r for r in range(6) if board[r][col] == 0)
        temp = board.copy(); temp[row][col] = piece
        score = minimax(temp, depth-1, -math.inf, math.inf, False, piece)
        if score > best_score:
            best_score, best_col = score, col
    return best_col


# ── Training loop ──────────────────────────────────────────────────────────────

def run_training():
    env   = Connect4Env()
    agent = RLAgent()   # DQN agent, plays as piece 2

    for ep in range(1, NUM_EPISODES + 1):
        state = env.reset()
        turn  = 1        # piece 1 (minimax opponent) goes first
        prev_state  = None
        prev_action = None

        while not env.game_over:
            valid = env.get_valid_locations()
            if not valid:
                # Draw
                if prev_state is not None:
                    agent.train_step(prev_state, prev_action, R_DRAW,
                                     env.board.copy(), True)
                agent.end_episode("draw")
                break

            # ── Opponent turn (Minimax, piece 1) ──────────────────────────
            if turn == 1:
                col = minimax_action(env.board, OPPONENT_DEPTH, piece=1)
                row = env.get_next_open_row(col)
                env.drop_piece(row, col, 1)

                if env.winning_move(1):
                    env.game_over = True
                    # The agent's last move led to this loss
                    if prev_state is not None:
                        agent.train_step(prev_state, prev_action, R_LOSS,
                                         env.board.copy(), True)
                    agent.end_episode("loss")
                    break

                turn = 2

            # ── Agent turn (DQN, piece 2) ──────────────────────────────────
            else:
                _, features = env.get_reward_and_features(piece=2)
                col         = agent.get_action(env.board.copy(), valid, features)
                row         = env.get_next_open_row(col)

                board_before = env.board.copy()
                env.drop_piece(row, col, 2)
                board_after  = env.board.copy()

                if env.winning_move(2):
                    env.game_over = True
                    agent.train_step(board_before, col, R_WIN, board_after, True)
                    agent.end_episode("win")
                    break

                # Non-terminal: store this transition; we'll update it next
                # agent turn with the real next-state (after opponent replies)
                if prev_state is not None:
                    agent.train_step(prev_state, prev_action, 0.0,
                                     board_after, False)

                prev_state  = board_before
                prev_action = col
                turn = 1

    # Final save
    agent.save_q_table()
    print("\nTraining complete.")
    print(f"  W={agent.wins}  L={agent.losses}  D={agent.draws}")
    print(f"  Replay buffer size: {len(agent.replay)}")


if __name__ == "__main__":
    run_training()