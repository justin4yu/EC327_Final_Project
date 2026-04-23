# agent.py
import random
import pickle
import os
import numpy as np

class RLAgent:
    """
    Tabular Q-Learning Agent for Connect 4.

    STATE REPRESENTATION
    --------------------
    The full 6x7 board is too large for a naive table (~3^42 states), so we
    compress it into a compact tuple key that is still fast to look up:
        - The raw board flattened to a tuple of ints (0/1/2).
    In practice only a small fraction of those states are ever visited, so the
    defaultdict stays manageable during a training session.

    Q-TABLE
    -------
    q_table: dict  { state_key -> np.array of shape (COLS,) }
    Each entry stores the estimated future reward for dropping in that column.
    Columns that are full are masked to -inf before argmax so the agent never
    tries an illegal move.

    HYPERPARAMETERS (all exposed for easy student experimentation)
    --------------------------------------------------------------
    alpha       : learning rate           (how fast we update Q values)
    gamma       : discount factor         (how much we value future rewards)
    epsilon     : exploration rate        (probability of random move)
    epsilon_min : floor on epsilon        (always keep some exploration)
    epsilon_decay: multiplicative decay applied after every episode
    """

    COLS = 7

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        q_table_path: str = "q_table.pkl",
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table_path  = q_table_path

        # Load a previously saved table if one exists, otherwise start fresh
        self.q_table: dict = self._load_q_table()

        # ---- training bookkeeping ----
        self.episode_count = 0
        self.wins   = 0
        self.losses = 0
        self.draws  = 0

    # ------------------------------------------------------------------
    # Public interface expected by connect4_engine.py
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray, valid_moves: list, info: dict) -> int:
        """
        Epsilon-greedy action selection.

        During TRAINING  -> call this, then call train_step() with the result.
        During INFERENCE -> set self.epsilon = 0 before calling this.

        Parameters
        ----------
        state       : 6×7 numpy array (0=empty, 1=human, 2=agent)
        valid_moves : list of legal column indices
        info        : feature dict from the engine (available for shaped rewards,
                      logging, or future use in a neural-network version)

        Returns
        -------
        col : int  — column index to drop the piece into
        """
        print(f"[Q-Agent] ε={self.epsilon:.3f}  features={info}")

        # ε-greedy: explore
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # exploit: pick the column with the highest Q-value among legal moves
        state_key = self._state_to_key(state)
        q_values  = self._get_q_values(state_key)

        # Mask illegal columns
        masked = np.full(self.COLS, -np.inf)
        for col in valid_moves:
            masked[col] = q_values[col]

        return int(np.argmax(masked))

    def train_step(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        next_valid_moves: list = None,
    ) -> float:
        """
        One Q-learning update (off-policy TD(0)):

            Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]

        Call this every step inside your headless training loop.

        Parameters
        ----------
        state            : board before the move
        action           : column chosen
        reward           : scalar reward received
        next_state       : board after the move
        done             : True if the game ended
        next_valid_moves : legal columns in next_state (optional; defaults to all)

        Returns
        -------
        td_error : float  — useful for logging / debugging
        """
        s_key  = self._state_to_key(state)
        s1_key = self._state_to_key(next_state)

        current_q = self._get_q_values(s_key)[action]

        if done:
            target = reward
        else:
            next_q_values = self._get_q_values(s1_key)
            if next_valid_moves is None:
                next_valid_moves = list(range(self.COLS))
            best_next = max(next_q_values[col] for col in next_valid_moves)
            target = reward + self.gamma * best_next

        td_error = target - current_q
        self._get_q_values(s_key)[action] += self.alpha * td_error
        return td_error

    def end_episode(self, outcome: str = "draw"):
        """
        Call once at the end of every training game.
        Handles epsilon decay and bookkeeping.

        outcome : "win" | "loss" | "draw"
        """
        self.episode_count += 1
        if outcome == "win":
            self.wins   += 1
        elif outcome == "loss":
            self.losses += 1
        else:
            self.draws  += 1

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodic checkpoint
        if self.episode_count % 500 == 0:
            self.save_q_table()
            print(
                f"[Q-Agent] ep={self.episode_count:>6}  "
                f"W/L/D={self.wins}/{self.losses}/{self.draws}  "
                f"ε={self.epsilon:.4f}  "
                f"|Q|={len(self.q_table)}"
            )

    def save_q_table(self):
        with open(self.q_table_path, "wb") as f:
            pickle.dump(self.q_table, f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_to_key(self, board: np.ndarray) -> tuple:
        """Flatten the board to a hashable tuple (fast dict key)."""
        return tuple(board.astype(int).flatten())

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Return (and lazily initialise) the Q-value vector for a state."""
        if state_key not in self.q_table:
            # Optimistic initialisation: small positive values encourage exploration
            self.q_table[state_key] = np.zeros(self.COLS)
        return self.q_table[state_key]

    def _load_q_table(self) -> dict:
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, "rb") as f:
                    table = pickle.load(f)
                print(f"[Q-Agent] Loaded Q-table with {len(table)} states from '{self.q_table_path}'")
                return table
            except (EOFError, pickle.UnpicklingError):
                print(f"[Q-Agent] Corrupted Q-table at '{self.q_table_path}' — starting fresh.")
                os.remove(self.q_table_path)
        print("[Q-Agent] No saved Q-table found — starting fresh.")
        return {}