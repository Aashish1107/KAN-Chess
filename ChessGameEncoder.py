import numpy as np

class ChessGameEncoder:
    def __init__(self):
        self.board=self.create_starting_position()
        self.move_vount=0
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
        
        return position
    def encode_position_to_tensor(self, board):
        #Generates a 8x8x119 numpy array/tensor representing the board position
        pass
    def visualize_channels(self, encoded_position):
        pass
    def validate_move(self, piece, fromPos, toPos):
        pass
    def decode_Command(self,command):
        pass
    def move_piece(self, command):
        piece, fromPos, toPos=self.decode_Command(command)
        isValid=self.validate_move(piece, fromPos, toPos)
        if not isValid:
            raise ValueError("Invalid Move")
        # Update board state
        self.board_history.append(self.board)
        while self.board_history.__len__()>7:
            self.board_history.pop(0)
        self.board["board"][toPos[0]][toPos[1]]=piece
        self.board["board"][fromPos[0]][fromPos[1]]=None
        self.move_vount+=1 if self.board["whiteToMove"]==1 else 0
        self.board["whiteToMove"]=1-self.board["whiteToMove"]
                    
if __name__ == "__main__":
    encoder = ChessGameEncoder()
    print(encoder.board)