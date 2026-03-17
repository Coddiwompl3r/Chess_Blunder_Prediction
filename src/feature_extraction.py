"""
feature_extraction.py
Phase 3: Extract measurable features from each chess position.

Each feature is a hypothesis about what causes human error.
The ML model will later tell us which hypotheses are correct.

Features extracted:
  - n_legal_moves      : branching factor (complexity hypothesis)
  - material_balance   : who is winning materially
  - n_pieces_on_board  : overall position density
  - is_endgame         : game phase classification
  - king_in_check      : tactical urgency
  - eval_volatility    : how unstable the position has been recently
  - prev_was_blunder   : the tilt hypothesis
  - prev_delta_cp      : momentum of recent play
  - player_rating      : player strength
  - move_number        : proxy for game phase
"""

import chess
import chess.pgn
import numpy as np
from collections import deque


def get_material_balance(board):
    """
    Calculate material balance from White's perspective.

    Piece values in centipawns (standard):
      Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900

    Positive = White has more material.
    Negative = Black has more material.
    """
    piece_values = {
        chess.PAWN:   100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK:   500,
        chess.QUEEN:  900,
        chess.KING:   0,
    }

    balance = 0
    for piece_type, value in piece_values.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        balance += value * (white_count - black_count)

    return balance


def count_pieces(board):
    """Count total pieces on the board."""
    return bin(board.occupied).count("1")


def is_endgame(board):
    """
    Classify position as endgame.
    Endgame = no queens on the board, OR fewer than 12 total pieces.
    """
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))

    if white_queens == 0 and black_queens == 0:
        return True

    if count_pieces(board) <= 12:
        return True

    return False


def get_game_phase(move_number):
    """Classify game phase by move number."""
    if move_number <= 10:
        return "opening"
    elif move_number <= 30:
        return "middlegame"
    else:
        return "endgame"


class EvaluationTracker:
    """
    Tracks recent evaluations to compute volatility.

    Volatility = standard deviation of the last N evaluations.
    High volatility means the position has been swinging wildly.
    Swinging positions = more blunder opportunities.
    """

    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def add(self, eval_cp):
        if eval_cp is not None:
            self.history.append(eval_cp)

    def get_volatility(self):
        if len(self.history) < 2:
            return 0.0
        return float(np.std(list(self.history)))


def extract_features(analyzed_moves, game, player_ratings):
    """
    Takes the output of analyze_game() and adds rich features
    to every move record.

    Parameters:
        analyzed_moves : list of dicts from engine_analysis.py
        game           : the chess.pgn.Game object
        player_ratings : {"white": int, "black": int}

    Returns:
        list of feature-enriched dictionaries, one per move
    """
    board         = game.board()
    eval_tracker  = EvaluationTracker(window_size=5)

    prev_was_blunder = False
    prev_delta_cp    = 0

    enriched = []

    for move_data in analyzed_moves:

        eval_tracker.add(move_data.get("eval_before_cp"))

        if move_data["player_to_move"] == "white":
            rating = player_ratings.get("white", 1500)
        else:
            rating = player_ratings.get("black", 1500)

        features = {
            # --- Position complexity ---
            "n_legal_moves":    move_data["n_legal_moves"],
            "n_pieces":         count_pieces(board),
            "material_balance": get_material_balance(board),
            "is_endgame":       int(is_endgame(board)),
            "king_in_check":    int(board.is_check()),

            # --- Evaluation features ---
            "eval_before_cp":   move_data.get("eval_before_cp") or 0,
            "eval_volatility":  eval_tracker.get_volatility(),
            "prev_delta_cp":    prev_delta_cp,

            # --- Game context ---
            "move_number":      move_data["move_number"],
            "game_phase":       get_game_phase(move_data["move_number"]),
            "player_to_move":   move_data["player_to_move"],
            "player_rating":    rating,

            # --- Tilt hypothesis ---
            "prev_was_blunder": int(prev_was_blunder),

            # --- Target variables ---
            "is_blunder":       int(move_data["is_blunder"]),
            "is_mistake":       int(move_data["is_mistake"]),
            "quality":          move_data["quality"],
            "delta_cp":         move_data.get("delta_cp") or 0,

            # --- Metadata ---
            "move_uci":         move_data["move_uci"],
            "fen_before":       move_data["fen_before"],
        }

        enriched.append(features)

        prev_was_blunder = move_data["is_blunder"]
        prev_delta_cp    = move_data.get("delta_cp") or 0

        try:
            board.push(chess.Move.from_uci(move_data["move_uci"]))
        except Exception:
            pass

    return enriched


def main():
    """Test feature extraction on the first sample game."""
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from engine_analysis import analyze_game, STOCKFISH_PATH
    import chess.engine

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    pgn_path    = os.path.join(project_dir, "data", "sample_games.pgn")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    with open(pgn_path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)

    print("Running engine analysis...")
    analyzed = analyze_game(game, engine, depth=10)

    ratings = {"white": 2882, "black": 2664}
    enriched = extract_features(analyzed, game, ratings)

    engine.quit()

    print(f"\nFeatures extracted for {len(enriched)} moves")
    print("\nSample — Move 1:")
    for key, value in enriched[0].items():
        print(f"  {key:<20}: {value}")

    print("\nSample — Move 20:")
    for key, value in enriched[19].items():
        print(f"  {key:<20}: {value}")

    print("\nPhase 3 complete. Feature extraction working.")


if __name__ == "__main__":
    main()


