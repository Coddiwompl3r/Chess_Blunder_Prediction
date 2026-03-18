"""
parse_lichess.py
Extracts clock data from Lichess PGN files.
"""

import chess
import chess.pgn
import os

def parse_time_control(time_control_str):
    """
    Extract starting seconds from a TimeControl string.
    "60+0"   → 60.0  (bullet: 1 minute, no increment)
    "600+5"  → 600.0 (rapid: 10 minutes, 5 sec increment)
    "-"      → None  (no time control listed)
    """
    try:
        parts = time_control_str.split("+")
        return float(parts[0])
    except:
        return None

def parse_game_with_clocks(game):
    records = []
    board = game.board()

    # Parse starting time from headers so move 1 has a valid prev_clock
    time_control_str = game.headers.get("TimeControl", "-")
    starting_seconds = parse_time_control(time_control_str)

    # Each player starts with the same time
    prev_clock_white = starting_seconds
    prev_clock_black = starting_seconds

    for node in game.mainline():
        move        = node.move
        clock_after = node.clock()

        # Select the correct player's previous clock
        if board.turn == chess.WHITE:
            prev_clock = prev_clock_white
        else:
            prev_clock = prev_clock_black

        # Calculate time spent on this move
        time_spent = None if prev_clock is None else prev_clock - clock_after
        time_pressure = None if (starting_seconds is None or starting_seconds == 0) else clock_after / starting_seconds

        record = {
            "move_uci":            move.uci(),
            "player_to_move":      "white" if board.turn == chess.WHITE else "black",
            "clock_after_seconds": clock_after,
            "time_spent_seconds":  time_spent,
            "time_pressure":    time_pressure,
        }

        records.append(record)

        # Capture who just moved BEFORE pushing
        turn_just_moved = board.turn
        board.push(move)

        # Update only the correct player's clock
        if turn_just_moved == chess.WHITE:
            prev_clock_white = clock_after
        else:
            prev_clock_black = clock_after

    return records

def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    pgn_path    = os.path.join(project_dir, "data", "lichess_sample.pgn")

    with open(pgn_path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)

    print(f"Game: {game.headers.get('White')} vs {game.headers.get('Black')}")
    print(f"Time control: {game.headers.get('TimeControl')}")

    records = parse_game_with_clocks(game)

    print(f"\nMoves parsed: {len(records)}")
    print("\nFirst 5 moves:")
    for r in records[:5]:
        print(f"  {r['player_to_move']:<6} | {r['move_uci']} | "
              f"clock: {r['clock_after_seconds']}s | "
              f"spent: {r['time_spent_seconds']}s | "
              f"pressure: {r['time_pressure']:.2f}")


if __name__ == "__main__":
    main()
