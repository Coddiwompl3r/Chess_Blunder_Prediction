"""
engine_analysis.py
Phase 2: Connect Stockfish to evaluate chess positions.

Every chess position becomes a number (centipawns).
That number is your scientific instrument for measuring human error.
"""

import chess
import chess.engine
import chess.pgn
import os

EVAL_CAP = 2000
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def evaluate_position(board, engine, depth=10):
    """
    Ask Stockfish to evaluate a position.
    Always returns score from WHITE's perspective.
    Positive = White is better. Negative = Black is better.
    Returns None if the game is already over.
    """
    if board.is_game_over():
        return None

    try:
        info  = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white().score(mate_score=9999)

        if score is not None:
            score = max(-EVAL_CAP, min(EVAL_CAP, score))

        return score

    except Exception as e:
        print(f"Engine error: {e}")
        return None


def classify_move(eval_before, eval_after, turn):
    """
    Classify a move as blunder/mistake/inaccuracy/good.

    CRITICAL: We measure the delta from the MOVING player's perspective.
    If White moves and evaluation drops, that is bad for White.
    If Black moves and evaluation rises (from White's view), that is bad for Black.
    """
    if eval_before is None or eval_after is None:
        return {"delta_cp": None, "quality": "unknown"}

    if turn == chess.WHITE:
        delta = eval_after - eval_before
    else:
        delta = eval_before - eval_after

    if delta <= -200:
        quality = "blunder"
    elif delta <= -100:
        quality = "mistake"
    elif delta <= -50:
        quality = "inaccuracy"
    else:
        quality = "good"

    return {"delta_cp": delta, "quality": quality}


def analyze_game(game, engine, depth=10):
    """
    Analyze every move in a game.
    Returns a list of dictionaries, one per move,
    containing the evaluation before and after each move
    and the move quality classification.
    """
    results = []
    board   = game.board()

    prev_eval = evaluate_position(board, engine, depth)

    move_number = 0

    for move in game.mainline_moves():
        move_number += 1

        turn_before  = board.turn
        fen_before   = board.fen()
        n_legal      = board.legal_moves.count()
        eval_before  = prev_eval

        board.push(move)

        eval_after   = evaluate_position(board, engine, depth)
        quality_info = classify_move(eval_before, eval_after, turn_before)

        record = {
            "move_number":    move_number,
            "player_to_move": "white" if turn_before == chess.WHITE else "black",
            "move_uci":       move.uci(),
            "fen_before":     fen_before,
            "n_legal_moves":  n_legal,
            "eval_before_cp": eval_before,
            "eval_after_cp":  eval_after,
            "delta_cp":       quality_info["delta_cp"],
            "quality":        quality_info["quality"],
            "is_blunder":     quality_info["quality"] == "blunder",
            "is_mistake":     quality_info["quality"] in ("blunder", "mistake"),
        }

        results.append(record)
        prev_eval = eval_after

    return results


def main():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    pgn_path    = os.path.join(project_dir, "data", "sample_games.pgn")

    with open(pgn_path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)

    white = game.headers.get("White", "?")
    black = game.headers.get("Black", "?")
    print(f"Analyzing: {white} vs {black}")
    print("Running Stockfish at depth 10 — takes about 30 seconds...\n")

    moves_data = analyze_game(game, engine, depth=10)

    total_moves   = len(moves_data)
    total_blunders = sum(1 for m in moves_data if m["is_blunder"])
    total_mistakes = sum(1 for m in moves_data if m["is_mistake"])

    print(f"Total moves analyzed : {total_moves}")
    print(f"Blunders detected    : {total_blunders}")
    print(f"Mistakes detected    : {total_mistakes}")
    print(f"Blunder rate         : {total_blunders / total_moves * 100:.1f}%")

    print("\n--- Sample output (moves 10 to 12) ---")
    for move in moves_data[9:12]:
        print(
            f"  Move {move['move_number']:>3} ({move['player_to_move']:<5}) | "
            f"{move['move_uci']} | "
            f"eval: {str(move['eval_before_cp']):>6} -> {str(move['eval_after_cp']):>6} | "
            f"delta: {str(move['delta_cp']):>6} | "
            f"{move['quality']}"
        )

    print("\n--- All blunders in this game ---")
    for move in moves_data:
        if move["is_blunder"]:
            print(
                f"  Move {move['move_number']:>3} ({move['player_to_move']:<5}) | "
                f"{move['move_uci']} | "
                f"delta: {move['delta_cp']} cp"
            )

    engine.quit()
    print("\nPhase 2 complete. Engine analysis working.")


if __name__ == "__main__":
    main()


