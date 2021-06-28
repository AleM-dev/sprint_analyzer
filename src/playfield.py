import numpy as np
import copy

from frame_rectangle import FrameRectangle

class Playfield:
    def __init__(self, df_playfield, scale):
        self.width = df_playfield['width']
        self.height = df_playfield['height']
        
        self.board = np.zeros((self.height, self.width)) #state of the playfield
        self.piece_board = np.zeros((4, self.width))
        self.filled_squares = 0
        self.frame_board = []

        playfield_min_x = df_playfield['min_x']*scale
        max_y = df_playfield['max_y']*scale
        square_size = df_playfield['square_size']*scale
        border = df_playfield['border_thickness']

        for y in range(self.height):
            min_y = int(max_y - (y+1)*square_size)
            row =  []
            for x in range(self.width):
                min_x = int(playfield_min_x + x*square_size)
                row.append(FrameRectangle(min_x + border, min_y + border, square_size - 2*border, square_size - 2*border))
            self.frame_board.append(row)
            
    def set_background_bottom(self, frame, max_row):
        for x in range(self.width):
            for y in range(max_row):
                self.frame_board[y][x].set_background(frame)

    def set_background_top(self, frame, min_row):
        print(range(min_row, self.height))
        for x in range(self.width):
                for y in range(min_row, self.height):
                    self.frame_board[y][x].set_background(frame)

    def set_current_frame(self, frame):
        for y in range(self.height):
            for x in range(self.width):
                self.frame_board[y][x].set_current_frame(frame)

    def inbound(self, piece, min_x):
        if min_x < 0:
            return False
        if min_x + len(piece[0]) > self.width:
            return False
        return True

    def fill_incorrect_squares(self, target):
        diff = self.board - target.board
        coords = np.argwhere((diff < 0))
        print(coords)
        for [y, x] in coords:
            self.board[y][x] = 1


    def board_diff(self, prev, num_pieces, ignore):
        #print(ignore)
        temp_board = self.board.copy()
        temp_board[temp_board > num_pieces] = 0
        temp_board[temp_board > 0] = 1
        temp_prev = prev.board.copy()
        temp_prev[temp_prev > num_pieces] = 0
        temp_prev[temp_prev > 0] = 1
        diff = temp_board - temp_prev
        new_squares = []
        for y in range(self.height):
            for x in range(self.width):
                if diff[y][x] == 1 and (x,y) not in ignore:
                    new_squares.append((x, y))
        return new_squares

    def update_with(self, pos, val, offset):
        for (x,y) in pos:
            self.board[y + offset][x] = val

    def print_board(self):
        board = self.board[-1,:]
        board = board[:,~np.all(board == 0, axis=0)]
        print(board)

    def reconstruct_board(self, df_pieces, prev, piece):
        matches = []
        print(piece)

            #piece_state = piece_state*piece['id']
        for r in range(len(piece['rotations'])):
            piece_state = np.array(piece['rotations'][r])
            for x in range(self.width):        
                if self.inbound(piece_state, x):
                    new_match = []
                    temp = copy.deepcopy(prev)
                    temp.hard_drop(piece_state, x, 1)
                    if self.match(temp.board, df_pieces['unknown'], False):
                        new_match.append((piece['id'], x, r))
                        matches.append(new_match)
        return matches

    def reconstruct_board_from_candidates(self, df_pieces, prev, prev_candidates, piece):
        matches = []
        print(piece)
            
        for i in range(len(prev_candidates)):
            current_path = prev_candidates[i]
            print('Current path', current_path)
            temp = copy.deepcopy(prev)
            for j in range(len(prev_candidates[i])):
                prev_id = prev_candidates[i][j][0]
                prev_x = prev_candidates[i][j][1]
                prev_rotation = prev_candidates[i][j][2]
                prev_piece = df_pieces['names'][prev_id]
                prev_state = np.array(df_pieces[prev_piece]['rotations'][prev_rotation])
            
                temp_j = copy.deepcopy(prev)
                            
                temp_j.hard_drop(prev_state, prev_x, 1)
                print(temp_j.board)

                for r in range(len(piece['rotations'])):
                    piece_state = np.array(piece['rotations'][r])
                    #piece_state = piece_state*piece['id']
                    for x in range(self.width):        
                        if self.inbound(piece_state, x):
                            new_match = current_path
                            temp = copy.deepcopy(temp_j)
                            temp.hard_drop(piece_state, x, 1)
                            if self.match(temp.board, df_pieces['unknown'], False):
                                print('Match at', piece['id'], x, r)
                                new_match.append((piece['id'], x, r))
                                matches.append(new_match)
        return matches

    def reconstruct_board_recursive(self, df_pieces, prev, i, path, candidates, matches):
        if i >= len(candidates):
            print('Adding', path)
            matches.append(path)
            return matches
        else:
            cand_i = candidates[i]
            for c in cand_i:
                piece_id = c[0]
                name = df_pieces['names'][piece_id]
                piece = df_pieces[name]
                x = c[1]
                rotation = c[2]
                print('Temp', c)

                piece_state = np.array(piece['rotations'][rotation])
                #piece_state = piece_state*piece['id']
                if self.inbound(piece_state, x):
                    temp = copy.deepcopy(prev)
                    temp.hard_drop(piece_state, x, piece_id)
                    print(temp.board)
                    if self.match(temp.board, df_pieces['unknown'], True):
                        print('Matched')
                        updated_path = copy.deepcopy(path)
                        updated_path.append((piece_id, x, rotation))
                        matches = self.reconstruct_board_recursive(df_pieces, temp, i+1, updated_path, candidates, matches)
            return matches

    def reconstruct_board_first_call(self, df_pieces, prev, candidates):
        print('Target')
        print(self.board)
        print('Candidates', candidates)
        matches = self.reconstruct_board_recursive(df_pieces, prev, 0, [], candidates, [])
        print('Reconstruction matches: ', matches)
        return matches
    

    # def reconstruct_board_from_candidates(self, df_pieces, prev, candidates):
    #     matches = []
                
    #     for i in range(len(candidates)):
    #         cand = candidates[i]
    #         piece_id = cand[0]
    #         name = df_pieces['names'][piece_id]
    #         piece = df_pieces[name]
    #         x = cand[1]
    #         rotation = cand[2]

    #         piece_state = np.array(piece['rotations'][rotation])
    #         #piece_state = piece_state*piece['id']
    #         if self.inbound(piece_state, x):
    #             temp = copy.deepcopy(prev)
    #             temp.hard_drop(piece_state, x, piece_id)
    #             if self.match(temp.board, df_pieces['unknown'], True):
    #                 matches.append((piece['id'], x, rotation))
    #     return matches


    def match(self, temp_board, unknown, sub_match): #test if all pieces in temp are in self.board
        temp_board = np.where(temp_board > 0, 1, 0)
        target_unknown = np.where(self.board == unknown, 2, 0)
        target_piece = np.where(self.board > 0, 1, 0)
        
        target = np.zeros_like(temp_board)
        target = target + target_piece
        target = target + target_unknown

        diff = target - temp_board
        #print(target)
        #print(temp_board)
#        print(diff)
        #print()
       
        if (diff >= 0).all():
            if sub_match:
                # print('Target')
                # print(target)
                # print('Board')
                # print(temp_board)
                return True
            elif not sub_match and (diff != 1).all() :
                # print('Target')
                # print(target)
                # print('Board')
                # print(temp_board)
                return True
        return False
    
    def hard_drop(self, piece, min_x, val):
        piece = piece[::-1] #flip in the vertical axis

        min_by_x = {}
        for x in range(len(piece[0])):
            min_by_x[x+min_x] = 3*self.height
        for y in range(len(piece)):
            for x in range(len(piece[0])):
                if piece[y][x] == 1:
                    min_y = min_by_x[x + min_x]
                    if y + 2*self.height < min_y:
                        min_by_x[x + min_x] = y + 2*self.height
        min_dist = 3*self.height
        for x in range(len(piece[0])):
            max_y = 0
            column = self.board[:, x + min_x]
            nonzero_indices = np.nonzero(column)[0]
            if nonzero_indices.size > 0:
                max_y = np.max(nonzero_indices) + 1

            if min_by_x[x + min_x] - max_y < min_dist:
                min_dist = min_by_x[x + min_x] - max_y

        for y in range(len(piece)):
            for x in range(len(piece[0])):
                if piece[y][x] > 0:
                    self.board[y + 2*self.height - min_dist][x + min_x] = val
        #print(board)

    def remove_effect(self, df):
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x] == df['pieces']['effect'] and y > df['playfield']['effect_space']:
                    remove = True
                    for dy in range(df['playfield']['effect_space']):
                        y2 = y - dy - 1
                        if self.board[y2][x] != df['pieces']['empty']:
                            remove = False
                    if remove:
                        self.board[y][x] = df['pieces']['empty']
    
    def clear_lines(self, cleared_lines):
        for y in reversed(cleared_lines):
            self.board = np.delete(self.board, y, axis=0)
        for _ in range(len(cleared_lines)):
            new_row = np.zeros(self.width)
            self.board = np.vstack([self.board, new_row])
        print(self.board)
