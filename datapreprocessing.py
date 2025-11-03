import pandas as pd
from ChessGameEncoder import ChessGameEncoder
from game import chessGame

# Read the CSV file and convert it into a pandas DataFrame
file_path = "data/games.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df["moves"].head())

df_moves = df[["white_rating", "black_rating", "moves"]].copy()
df_moves["min_rating"] = df_moves[["white_rating", "black_rating"]].min(axis=1)
df_moves = df_moves[df_moves["min_rating"] >= 2000]
df_moves=df_moves.where(df_moves["moves"].str.contains("#")).dropna()
df_moves = df_moves[["min_rating", "moves"]]

# Display the resulting DataFrame
print(df_moves.head())
#Only complete games
#print(df_moves["moves"].where(df_moves["moves"].str.contains("#")).dropna())
games=[[move for move in game.split(' ')] for game in df_moves["moves"]]
print(games[0])

encoder=ChessGameEncoder()
game=chessGame()
#games[0]=["e4","a5","e5","d5"]
for move in games[0]:
    game.move(move)
    board_state=game.report()
    encoder.new_state(board_state)
tensor=encoder.encode_position_to_tensor(encoder.board)
print(game.report())
encoder.visualize_channels(tensor, range(12))
#encoder.visualize_channels(tensor, 103)
#encoder.visualize_channels(tensor, [25,26,27])
#encoder.visualize_channels(tensor, [95,96,97,98,99])
print(game.result)