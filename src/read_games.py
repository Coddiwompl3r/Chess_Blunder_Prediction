import chess
import chess.pgn
import os

def load_games_from_pgn(filepath: str, max_games: int = 10):

games []
with open(filepath, encoding="utf-8") as pgn_file:
  game_count = 0

  while game_count < max_games:
    game = chess.pgn.read_game(pgn_file)

  if game is None:
    print(f"File ended. Read {game_count} games.")
    break

games.append(game)
game_count: += 1

print(f"Successful loaded {len(games)} games.")
return games

def extract_positions_from_game(game):
  positions = []
  board = game.board()


  white_player = game.headers.get("White", "Unknown")
  black_player = game.headers.get("Black", "Unknown")
  event        = game.headers.get("Event", "Unknown")
  result       = game.headers.get("Result", "*")

  move_number = 0

  for move in game.mailline_moves():

    move_number += 1
    position_data = {
      "event": event, 
      "white": white_player,
      "black": black_player, 
      "result": result, 
      "move_number": move_number, 
      "player_to_move": "white" if board.turn == chess.WHITE else "black", 
      "move_uci": move.uci(), 
      "fen": board.fen(),
      "n_legal_moves": board.legal_moves.count(), 
    }
    positions.append(position_data)

    board.push(move)

  return positions 

def main():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_dir = os.path.dirname(script_dir)
  pgn_path = os.path.join(project_dir, "data" , "sample_games.pgn")

  print(f"Loading games from: {pgn_path}")

  games = load_games_from_pgn(pgn_path, max_games=3)

  if not games:
    print("ERROR: No games loaded. Check your file path.")
    return

  print("\n--- Processing first game ---")
  positions = extract_postitions_from_game(games[0])

  print(f"Total moves in game: {len(positions)}")
  print("\nFirst move data:")

  for key, value in positions[0].items():
    print(f" {key}: {value}")

  print("\nLast move data")
  for key, value in positions[-1].items():
    print(f" {key}: {value}")

  print("\nPhase 1 complete. Board recontruction working.")

if __name__ == "__main__":
  main()





    
