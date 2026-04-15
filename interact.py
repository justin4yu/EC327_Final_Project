from playwright.sync_api import sync_playwright
import time

def run_rl_agent():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:3000") # Replace with your local dev URL

        while True:
            # 1. Get the current state
            game_data = page.evaluate("window.chess.state")
            if game_data['status'] in ['checkmate', 'stalemate']:
                print(f"Game Over: {game_data['status']}")
                page.evaluate("window.chess.reset()")
                continue

            # 2. Get legal moves
            legal_moves = page.evaluate("window.chess.getLegalMoves()")
            
            # 3. Simple Random Agent Logic (Replace with your RL Model)
            import random
            move = random.choice(legal_moves)
            fr, fc = move['from']
            tr, tc, _ = move['to']

            # 4. Execute move in the React app
            success = page.evaluate(f"window.chess.move({fr}, {fc}, {tr}, {tc})")
            
            if success:
                print(f"Moved from {fr},{fc} to {tr},{tc}")
            
            time.sleep(0.5) # Slow down for visibility

run_rl_agent()