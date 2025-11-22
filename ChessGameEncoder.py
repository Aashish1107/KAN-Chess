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
            "halfmoveClock": 0,
            "fullmoveNumber": 0
        }
        
        return position
    def encode_position_to_tensor(self, board):
        #Generates a 8x8x119 numpy array/tensor representing the board position
        tensor=np.zeros((8,8,119), dtype=np.float16)
        # Encode board pieces -12 channels
        self.encode_single_board(tensor, self.board)
        #Encode history -84 channels
        self.encode_board_history(tensor)
        #Encode castling rights - 4 channels
        self.encode_castling(tensor, self.board["castlingRights"])
        #Encode en passant target - 8 channel(100-107)
        self.encode_enpassant(tensor,self.board["enPassantTarget"])
        tensor [:,:,108]=self.board["halfmoveClock"]/100
        tensor [:,:,109]=self.board["whiteToMove"]
        #Encode repetition count
        #self.encode_repeatition(self.board)
        tensor[:,:,118]=self.board["fullmoveNumber"]/100.0
        
        return tensor
    
    def encode_board_history(self,tensor):
        for i,board in enumerate(reversed(self.board_history)):
            channelOffset=(i+1)*12
            self.encode_single_board(tensor,self.board_history[i], channelOffset)
            
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
        if enPassantTarget is not None:
            tensor[ :, : ,100+enPassantTarget]=1
        return tensor
    def encode_repeatition(self, board):
        for i in range(min(8,len(self.board_history))):
            board=self.board_history[-(i+1)]
            count=self.count_repeatitions(board)
            channel=110+i
            tensor[:,:,channel]=min(count,2)/2.0
    def count_repeatitions(self, board):
        count=0
        board_fen=self.board_to_fen(board)
        for past_board in self.board_history:
            if self.board_to_fen(past_board["board"])==board_fen:
                count+=1
        return count
    def reset_board(self):
        self.board_history=[]
        self.board=self.create_starting_position()
        self.move_count=0
        
    def visualize_channels(self, tensor, channels):
        if(type(channels)!=int and len(channels)>1 or channels=="all"):
            for c in channels:
                print("Channel: ", c)
                print(tensor[:,:,c])
        else:
            print("Channel: ", channels)
            print(tensor[:,:,channels])
    def new_state(self, board):
        # Update board state
        self.board_history.append(self.board)
        self.board=board
        while len(self.board_history)>7:
            self.board_history.pop(0)
        self.move_count+=1 if self.board["whiteToMove"]==1 else 0
        #self.board["whiteToMove"]=1-self.board["whiteToMove"]
                    
if __name__ == "__main__":
    encoder = ChessGameEncoder()
    tensor=encoder.encode_position_to_tensor(encoder.board)
    #encoder.visualize_channels(tensor, range(12))
    #encoder.visualize_channels(tensor, [95,96,97,98,99])