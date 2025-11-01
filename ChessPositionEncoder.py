import numpy as np

class ChessGameEncoder:
    def __init__(self):
        self.board=self.create_starting_position()
        self.board_history=[]
        self.piece_types = {'P':1, 'N':3, 'B':3.1, 'R':5, 'Q':9, 'K':100, 'p':1, 'n':3, 'b':3.1, 'r':5, 'q':9, 'k':100}
    def create_starting_position(self):
        position={
            "board":[['r','n','b','q','k','b','n','r'],
            ['p','p','p','p','p','p','p','p'],
            [None]*8,
            [None]*8,
            [None]*8,
            [None]*8,
            ['P','P','P','P','P','P','P','P'],
            ['R','N','B','Q','K','B','N','R']],
            "whiteToMove":1,
            "castlingRights":{'K':True,'Q':True,'k':True,'q':True},
            "enPassantTarget":None,
        }
        self.board_history=[position]
        return position
    def encode_position(self, board):
        pass
    def visualize_channels(self, encoded_position):
        pass
    def decode_Command(self,command):
        pass
    def move_piece(self, piece, command):
        pass
            
if __name__ == "__main__":
    encoder = ChessGameEncoder()
    print(encoder.board)