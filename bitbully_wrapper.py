import bitbully as bb
import os
import sys
from contextlib import contextmanager

# Initialize the solver once
agent = bb.BitBully()

@contextmanager
def silence_stdout():
    """Redirects stdout/stderr to devnull to hide library-internal error prints."""
    new_target = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = new_target
    sys.stderr = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        new_target.close()

def get_bitbully_score(move_history):
    """
    Evaluates the board using BitBully. 
    Returns the score from the perspective of the player whose turn it is.
    """
    move_str = "".join(str(m + 1) for m in move_history)
    
    # We use the silence manager to prevent the library from 
    # flooding your Colab console with 'Unexpected error' messages.
    with silence_stdout():
        try:
            board = bb.Board(move_str)
            # Use mtdf search to get the exact value
            score = agent.mtdf(board)
            return float(score)
        except Exception:
            # If the board is invalid (already won or full), return 0
            return 0.0