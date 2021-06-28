import numpy as np
import cv2
import scipy.misc
import reader
import copy

from frame_rectangle import FrameRectangle
from drawing import Draw
from reader import Reader
from session import SessionStats, TimingStats, PieceStats
from playfield import Playfield
from frame_processing import FrameProcessing
from timer import Time

def flip_piece(piece):
    reversed = piece[::-1]
    return reversed

def reconstruct_board(board, target, piece):
    matches = []
    width = board.shape[1]
    print( len(piece['rotations']) )
    
    for i in range(len(piece['rotations'])):
        piece_state = np.array(piece['rotations'][i])
        piece_state = piece_state*piece['id']
        for x in range(width):        
            if inbound(piece_state, x, width):
                temp_board = board.copy()
                temp_board = hard_drop(temp_board, piece_state, x, board.shape[0])
                print(temp_board)
                if board_sub_match(temp_board, target, 9):
                    matches.append((x, i))
    return matches

def inbound(piece, min_x, width):
    if min_x < 0:
        return False
    if min_x + len(piece[0]) > width:
        return False
    return True

def board_sub_match(board, target, max_val): #test if all pieces in board are in target
    diff = target - board
    exact_match = np.where(diff == 0, 1, 0)
    unknown_match = np.where(diff > max_val, 1, 0)
    bin_match = exact_match + unknown_match

    board_bin = np.where(board >= 1, 1, 0)
    match = np.multiply(board_bin, bin_match)

    if np.array_equal(board_bin, match):
        return True
    else:
        return False

def hard_drop(board, piece, min_x, height):
    piece = flip_piece(piece)

    min_by_x = {}
    for x in range(len(piece[0])):
        min_by_x[x+min_x] = 3*height
    for y in range(len(piece)):
        for x in range(len(piece[0])):
            if piece[y][x] == 1:
                min_y = min_by_x[x + min_x]
                if y + 2*height < min_y:
                    min_by_x[x + min_x] = y + 2*height
    min_dist = 3*height
    for x in range(len(piece[0])):
        max_y = 0
        column = board[:, x + min_x]
        nonzero_indices = np.nonzero(column)[0]
        if nonzero_indices.size > 0:
            max_y = np.max(nonzero_indices) + 1

        if min_by_x[x + min_x] - max_y < min_dist:
            min_dist = min_by_x[x + min_x] - max_y

    for y in range(len(piece)):
        for x in range(len(piece[0])):
            if piece[y][x] > 0:
                board[y + 2*height - min_dist][x + min_x] = piece[y][x]
    #print(board)
    return board

df = Reader.read_yaml()

width = df['playfield']['width']
height = df['playfield']['height']
        
board = np.zeros((height, width))

#i_piece = np.array(df['pieces']['i_piece']['rotations'][0])
i_piece = df['pieces']['i_piece']
j_piece = df['pieces']['j_piece']

#board = hard_drop(board, i_state, 0, height)

target_list = [
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
target = np.array(target_list)

candidates = reconstruct_board(board, target, i_piece)

print(candidates)
x = candidates[0][0]
i = candidates[0][1]
board = hard_drop(board, np.array(i_piece['rotations'][i]), x, board.shape[0])
print(board)

candidates = reconstruct_board(board, target, j_piece)
print(candidates)


board_2 = np.zeros((height, width))
board_2.fill(100)
#print(board_sub_match(board, board_2, df['pieces']['max_val']))