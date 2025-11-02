import numpy as np

piece_types = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5, 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}

class ChessGameEncoder:
    def __init__(self):
        self.board=self.create_starting_position()
        self.move_count=0
        self.board_history=[]
        
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
        tensor=np.zeros((8,8,119), dtype=np.uint8)
        # Encode board pieces -12 channels
        self.encode_single_board(tensor, self.board)
        #Encode history -84 channels
        self.encode_board_history(tensor)
        #Encode castling rights - 4 channels
        self.encode_castling(tensor, self.board["castlingRights"])
        #Encode en passant target - 8 channel
        #tensor+=self.encode_enpassant(self.board["enPassantTarget"])
        #Encode repetition count
        #tensor+=self.encode_repeatition(self.board)
        
        return tensor
    
    def encode_board_history(self,tensor):
        for i,board in enumerate(reversed(self.board_history)):
            channelOffset=(i+1)*12
            self.encode_single_board(self.board_history[i], channelOffset)
            
    def encode_single_board(self, tensor, board, channelOffset=0):
        for r in range(8):
            for c in range(8):
                piece=board["board"][r][c]
                if piece is not None:
                    piece_index=channelOffset+piece_types[piece]
                    tensor[r,c,piece_index]=1
                    
    def encode_castling(self, tensor, castlingRights):
        if castlingRights['K']:
            tensor[:,:,96]=1
        if castlingRights['Q']:
            tensor[:,:,97]=1
        if castlingRights['k']:
            tensor[:,:,98]=1
        if castlingRights['q']:
            tensor[:,:,99]=1
    def encode_enpassant(self, tensor, enPassantTarget):
        pass
    def encode_repeatition(self, board):
        pass
    def count_repeatitions(self, board):
        pass
    def reset_board(self):
        self.board_history=[]
        self.board=self.create_starting_position()
        self.move_vount=0
        
    def visualize_channels(self, tensor, channels):
        if(len(channels)>1 or channels=="all"):
            for c in channels:
                print("Channel: ", c)
                print(tensor[:,:,c])
        else:
            print(tensor[:,:,channels])
    def new_state(self, board):
        # Update board state
        self.board_history.append(self.board)
        self.board=board
        while len(self.board_history)>7:
            self.board_history.pop(0)
        self.move_count+=1 if self.board["whiteToMove"]==1 else 0
        self.board["whiteToMove"]=1-self.board["whiteToMove"]
                    
if __name__ == "__main__":
    encoder = ChessGameEncoder()
    tensor=encoder.encode_position_to_tensor(encoder.board)
    #encoder.visualize_channels(tensor, range(12))
    #encoder.visualize_channels(tensor, [95,96,97,98,99])