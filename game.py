import chess
import numpy as np

def convert_board_to_array(board):
    board_array=np.full((8,8),None)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        board_array[7 - row][col] = piece.symbol()
    return board_array

class chessGame:
    def __init__(self):
        self.board = chess.Board()
        result="0-0"

    def print_board(self):
        print(self.board)

    def convert_board_to_array(self):
        return convert_board_to_array(self.board)

    def move(self, move):
        try:
            chess_move=self.board.parse_san(move)
            if chess_move in self.board.legal_moves:
                self.board.push(chess_move)
                return True
            else:
                return False
        except ValueError:
            return False
    def report(self):
        return {
            "board": self.convert_board_to_array(),
            "whiteToMove": 1 if self.board.turn==chess.WHITE else 0,
            "castlingRights": {
                'K': self.board.has_kingside_castling_rights(chess.WHITE),
                'Q': self.board.has_queenside_castling_rights(chess.WHITE),
                'k': self.board.has_kingside_castling_rights(chess.BLACK),
                'q': self.board.has_queenside_castling_rights(chess.BLACK),
            },
            "enPassantTarget": None
        }
    
def newGame(command=""):
    game=chessGame()
    while(command!="exit"):
        if game.board.is_stalemate() or game.board.is_checkmate() or game.board.is_insufficient_material():
            game.result="1-0" if game.board.is_checkmate() and game.board.turn==chess.BLACK else "0-1" if game.board.is_checkmate() and game.board.turn==chess.WHITE else "1/2-1/2"
            print(f"Game over! Result: {game.result}")
            yield (game.report(), True)
            return
        command= yield (game.report(), False)
        command=str(command)
        if command=="exit":
            game.result="1-0" if game.board.turn==chess.BLACK else "0-1"
            user="White" if game.board.turn==chess.WHITE else "Black"
            print("Game exited by "+user)
            break
        if(command=="draw"):
            game.result="1/2-1/2"
            print("Game drawn by agreement.")
            break
        if game.move(command):
            print("Move accepted.")
        else:
            print("Illegal move. Try again.")

if __name__ == "__main__":
    game=newGame()
    report=next(game)
    while True:
        print(report)
        command=str(input("Enter Move:"))
        try:
            report, end=game.send(command)
        except StopIteration as e:
            print("Exiting Game.")
            break
        if command=="exit" or end:
            break
    print("Final Board Position:")
    print(report)