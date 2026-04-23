# train.py
"""
Headless self-play training loop for the Q-Learning Connect 4 agent.

Run:
    python train.py

The script pits Agent-2 (the learner) against Agent-1 (a copy that also
learns — self-play), so both players improve together. After training you can
play against the saved Q-table by running:
    python connect4_engine.py
"""

import numpy as np
from connect4_engine import Connect4Env
from agent import RLAgent

# ── Reward shaping constants ─────────────────────────────────────────────────
R_WIN          =  1.0    # agent wins the game
R_LOSS         = -1.0    # agent loses
R_DRAW         =  0.5    # board is full, no winner
R_HEURISTIC    =  0.01   # weight applied to engine's heuristic delta each step

# ── Training config ───────────────────────────────────────────────────────────
NUM_EPISODES   = 200_000  # total self-play games
PRINT_EVERY    = 500     # console log frequency (also triggers a checkpoint)


def shaped_reward(env: Connect4Env, piece: int) -> float:
    """Small intermediate reward derived from the engine's board heuristic."""
    score, _ = env.get_reward_and_features(piece)
    return score * R_HEURISTIC


def run_training():
    env      = Connect4Env()
    # Two independent Q-agents: agent1 plays piece=1, agent2 plays piece=2
    # agent2 is the one we ultimately save and use in the UI
    agent1   = RLAgent(q_table_path="q_table_p1.pkl")
    agent2   = RLAgent(q_table_path="q_table.pkl")   # <── used by the engine UI

    agents   = {1: agent1, 2: agent2}
    pieces   = {1: 1,      2: 2}

    for ep in range(1, NUM_EPISODES + 1):
        state = env.reset()
        prev  = {1: None, 2: None}   # stores (state, action) for delayed update
        turn  = 1                     # piece 1 goes first

        while not env.game_over:
            agent       = agents[turn]
            piece       = pieces[turn]
            valid_moves = env.get_valid_locations()

            if not valid_moves:
                # Board is full → draw
                for p in (1, 2):
                    if prev[p] is not None:
                        agents[p].train_step(*prev[p], R_DRAW, env.board.copy(), True, [])
                    agents[p].end_episode("draw")
                break

            _, features = env.get_reward_and_features(piece)
            action      = agent.get_action(state.copy(), valid_moves, features)

            row = env.get_next_open_row(action)
            env.drop_piece(row, action, piece)
            next_state = env.board.copy()

            # ── check for win ─────────────────────────────────────────────
            if env.winning_move(piece):
                env.game_over = True
                winner_piece  = piece
                loser_piece   = 2 if piece == 1 else 1

                # Update the winner's last transition with +WIN reward
                if prev[piece] is not None:
                    agents[piece].train_step(
                        *prev[piece], R_WIN, next_state, True, []
                    )
                else:
                    # First move happened to win (very rare)
                    agents[piece].train_step(
                        state.copy(), action, R_WIN, next_state, True, []
                    )

                # Update the loser's last transition with -LOSS reward
                if prev[loser_piece] is not None:
                    agents[loser_piece].train_step(
                        *prev[loser_piece], R_LOSS, next_state, True, []
                    )

                agents[piece].end_episode("win")
                agents[loser_piece].end_episode("loss")
                break

            # ── check for draw after this move (board just became full) ────
            next_valid_moves = env.get_valid_locations()
            if not next_valid_moves:
                env.game_over = True
                other_piece = 2 if piece == 1 else 1

                if prev[piece] is not None:
                    agents[piece].train_step(*prev[piece], R_DRAW, next_state, True, [])
                agents[piece].train_step(state.copy(), action, R_DRAW, next_state, True, [])

                if prev[other_piece] is not None:
                    agents[other_piece].train_step(*prev[other_piece], R_DRAW, next_state, True, [])

                agents[piece].end_episode("draw")
                agents[other_piece].end_episode("draw")
                break

            # ── mid-game: update the *previous* player's transition ───────
            # We can only give a meaningful next-state reward once we know
            # the opponent has moved and not won, so we do a one-step delay.
            step_reward = shaped_reward(env, piece)

            if prev[piece] is not None:
                agents[piece].train_step(
                    *prev[piece],
                    step_reward,
                    next_state.copy(),
                    False,
                    next_valid_moves,
                )

            prev[piece]  = (state.copy(), action)
            state        = next_state
            turn         = 2 if turn == 1 else 1   # swap turns

    # Final save
    agent2.save_q_table()
    agent1.save_q_table()
    print("\n✓ Training complete. Q-tables saved.")
    print(f"  agent2 (UI opponent):  W={agent2.wins}  L={agent2.losses}  D={agent2.draws}")
    print(f"  |Q-table| = {len(agent2.q_table)} unique states visited")


if __name__ == "__main__":
    run_training()