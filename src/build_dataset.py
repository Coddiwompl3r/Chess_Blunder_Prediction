"""
build_dataset.py
Phase 4: Run the full pipeline on multiple games and save a CSV dataset.

Pipeline:
  PGN file -> engine analysis -> feature extraction -> CSV

Usage:
    python3 src/build_dataset.py

Output:
    data/chess_blunder_dataset.csv
"""

import chess.pgn
import chess.engine
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine_analysis    import analyze_game, STOCKFISH_PATH
from feature_extraction import extract_features


def get_player_ratings(game):
    """
    Extract player ratings from PGN headers.
    Returns 1500 as default if rating is missing
    (1500 is the standard starting rating on Lichess).
    """
    try:
        white_rating = int(game.headers.get("WhiteElo", 1500))
    except (ValueError, TypeError):
        white_rating = 1500

    try:
        black_rating = int(game.headers.get("BlackElo", 1500))
    except (ValueError, TypeError):
        black_rating = 1500

    return {"white": white_rating, "black": black_rating}


def build_dataset(pgn_path, output_path, max_games=5, depth=10):
    """
    Process multiple games and save results to CSV.

    Parameters:
        pgn_path    : path to your PGN file
        output_path : where to save the dataset
        max_games   : how many games to process
        depth       : Stockfish search depth
    """
    print(f"Building dataset from: {pgn_path}")
    print(f"Games to process : {max_games}")
    print(f"Engine depth     : {depth}")
    print(f"Estimated time   : {max_games * 20}–{max_games * 40} seconds\n")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    all_records = []
    games_processed = 0
    games_failed    = 0

    with open(pgn_path, encoding="utf-8") as pgn_file:

        for game_idx in range(max_games):

            game = chess.pgn.read_game(pgn_file)
            if game is None:
                print(f"File ended at game {game_idx}.")
                break

            white = game.headers.get("White", "Unknown")
            black = game.headers.get("Black", "Unknown")
            print(f"Game {game_idx + 1}/{max_games}: {white} vs {black}")

            try:
                # Step 1: Engine analysis
                analyzed = analyze_game(game, engine, depth=depth)

                # Step 2: Feature extraction
                ratings  = get_player_ratings(game)
                features = extract_features(analyzed, game, ratings)

                # Step 3: Add game-level metadata to every move record
                for record in features:
                    record["game_id"]    = game_idx
                    record["white_name"] = white
                    record["black_name"] = black
                    record["event"]      = game.headers.get("Event",  "Unknown")
                    record["game_date"]  = game.headers.get("Date",   "Unknown")
                    record["game_result"]= game.headers.get("Result", "*")

                all_records.extend(features)
                games_processed += 1

                blunders = sum(1 for r in features if r["is_blunder"])
                mistakes = sum(1 for r in features if r["is_mistake"])
                print(f"  -> {len(features)} moves | {blunders} blunders | {mistakes} mistakes")

            except Exception as e:
                print(f"  ERROR on game {game_idx}: {e}")
                games_failed += 1
                continue

    engine.quit()

    if not all_records:
        print("\nERROR: No records collected. Check your PGN file.")
        return None

    df = pd.DataFrame(all_records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"DATASET COMPLETE")
    print(f"{'='*50}")
    print(f"Games processed : {games_processed}")
    print(f"Games failed    : {games_failed}")
    print(f"Total moves     : {len(df)}")
    print(f"Total blunders  : {df['is_blunder'].sum()}")
    print(f"Total mistakes  : {df['is_mistake'].sum()}")
    print(f"Blunder rate    : {df['is_blunder'].mean() * 100:.2f}%")
    print(f"Mistake rate    : {df['is_mistake'].mean() * 100:.2f}%")
    print(f"\nColumns in dataset ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col}")
    print(f"\nSaved to: {output_path}")

    return df


if __name__ == "__main__":

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    pgn_path    = os.path.join(project_dir, "data", "sample_games.pgn")
    output_path = os.path.join(project_dir, "data", "chess_blunder_dataset.csv")

    df = build_dataset(
        pgn_path    = pgn_path,
        output_path = output_path,
        max_games   = 5,
        depth       = 10
    )


