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
from piece import Piece

def init_hold(defaults, scale):
    min_x = int(defaults['min_x']*scale)
    min_y = int(defaults['min_y']*scale)
    width = int(defaults['square_size']*defaults['width']*scale) 
    height = int(defaults['square_size']*defaults['height']*scale)

    return FrameRectangle(min_x, min_y, width, height)

def init_pieces(df):
    pieces = []
    names = df['pieces']['names']
    for i in range(df['sprint']['num_pieces']):
        name = names[i + 1]
        piece = Piece(i + 1, df['pieces'][name]['rotations'][0])
        pieces.append(piece)
    return pieces

def set_piece_color(df, pieces, next):
    piece_id = FrameProcessing.map_frame_to_piece(df, next.current_frame, pieces)
    binary = binary = FrameProcessing.color_to_binary(next.current_frame, df['threshold']['piece_intensity'])
    if piece_id != -1 and not pieces[piece_id-1].color_defined:
        piece = pieces[piece_id-1]
        piece.color = FrameProcessing.average_color(next.current_frame, binary)
#        print(piece.id, piece.color)
        piece.color_defined = True
        return True
    else:
        return False
    
def init_nexts(defaults, scale):
    nexts = []

    min_x = int(defaults['next_1']['min_x']*scale)
    min_y = int(defaults['next_1']['min_y']*scale)
    width = int(defaults['next_1']['square_size']*defaults['next_1']['width']*scale) 
    height = int(defaults['next_1']['square_size']*defaults['next_1']['height']*scale)

    nexts.append(FrameRectangle(min_x, min_y, width, height))

    for y in range(defaults['next_n']['previews']):
        min_x = int(defaults['next_n']['min_x']*scale)
        min_y = int((defaults['next_n']['min_y'] + y*defaults['next_n']['vertical_gap'])*scale)
        width = int(defaults['next_n']['square_size']*defaults['next_n']['width']*scale) 
        height = int(defaults['next_n']['square_size']*defaults['next_n']['height']*scale)

        nexts.append(FrameRectangle(min_x, min_y, width, height))
    return nexts

def init_frame_digits(timer, scale):
    frame_digits = []
    for i in range(timer['digits']):
        min_x = int(timer['min_xs'][i] * scale)
        min_y = int(timer['min_y'] * scale)
        width = int(timer['width']*scale)
        height = int(timer['height']*scale)

        frame_digits.append(FrameRectangle(min_x, min_y, width, height))

    return frame_digits

def init_ignore_squares(df):
    ignore = []
    for i in range(len(df['overlay']['ignore_squares'])):
        ign = df['overlay']['ignore_squares']
        ignore.append((ign[i][0], ign[i][1]))
    return ignore

def update_ignore_tetris(df_overlay, ignore, width, height, line, expiry):
    min_y = df_overlay['tetris_min_y']
    max_y = df_overlay['tetris_max_y']

    for y in range(line + min_y, line + max_y + 1):
        for x in range(df_overlay['tetris_min_x'], df_overlay['tetris_max_x']+1):
            ignore[(x, y)] = expiry

    return ignore

def remove_from_ignore(ignore, frame_count):
    delete = []
    for key in ignore:
        #print(key, ignore[key])
        if ignore[key] <= frame_count:
            delete.append(key)
    
    for key in delete: 
        del ignore[key] 
    return ignore

def store_number(numbers, frame_rect, threshold):
    binary = FrameProcessing.color_to_binary(frame_rect.current_frame, threshold)
    numbers.append(binary)
    #print(binary)
    return numbers

def set_current_frames(df, frame, playfield, hold, nexts, frame_digits):
    playfield.set_current_frame(frame)
    if hold is not None:
        hold.set_current_frame(frame)
    if nexts is not None:
        for i in range(len(nexts)):
            nexts[i].set_current_frame(frame)
            
            

    for i in range(len(frame_digits)):
        frame_digits[i].set_current_frame(frame)

def timer_is_zero(frame_digits, df_threshold):
    last_digit = FrameProcessing.color_to_binary(frame_digits[len(frame_digits)-1].current_frame, df_threshold['digit'])
    area_ratio = np.sum(last_digit)/last_digit.size
    min = df_threshold['zero_min_area']
    max = df_threshold['zero_max_area']
    if area_ratio >= min and area_ratio <= max and timer_digits_match(frame_digits, df_threshold):
        return True
    else:
        return False

def timer_digits_match(frame_digits, df_threshold):
    count_matches = 0
    last_digit = FrameProcessing.color_to_binary(frame_digits[len(frame_digits)-1].current_frame, df_threshold['digit'])
    for i in range(0, len(frame_digits)-1):
        frame_i = FrameProcessing.color_to_binary(frame_digits[i].current_frame, df_threshold['digit'])
        score = FrameProcessing.compute_match(last_digit, frame_i)
        if score >= df_threshold['match']:
            count_matches += 1
        if count_matches == 4:
            return True
    return False
        
def match_digits(frame_digits, numbers, df_threshold):
    digits = []
    for i in range(len(frame_digits)):
        max_j = 0
        max_score = 0
        for j in range(len(numbers)):
            digit = FrameProcessing.color_to_binary(frame_digits[i].current_frame, df_threshold['digit'])
            number = numbers[j]
            score = FrameProcessing.compute_match(digit, number)
            if score > max_score:
                max_score = score
                max_j = j
        digits.append(max_j)
    return digits

def nexts_changed(defaults, nexts):
    count = 0
    if nexts is not None:
        for i in range(len(nexts)):
            diff = nexts[i].previous_frame_diff()
            if diff is not None:
                avg = np.average(diff)
                if avg > defaults['piece_diff']:
                    count += 1
        print("------ Testing if Nexts changed here...", count)
        if count > defaults['next_count']:
                
            return True
    return False                


def hold_changed(defaults, hold, session):
    diff = hold.previous_frame_diff()
    if diff is not None:
        avg = np.average(diff)
        print('Hold diff avg', avg)
        if avg > defaults['piece_diff']:
            return True
    return False      

def line_cleared(df, playfield):
    cleared = []

    for y in range(playfield.height):
        count_white = 0
        for x in range(10):
            sq1 = playfield.frame_board[y][x]
            diff1 = sq1.background_diff()
            sq2 = playfield.frame_board[y][x]
            diff2 = sq2.background_diff()
            
            avgs1 = [np.average(sq1.current_frame[:,:,0]), np.average(sq1.current_frame[:,:,1]), np.average(sq1.current_frame[:,:,2])]
            std1 = np.std(avgs1)
            avgs2 = [np.average(sq2.current_frame[:,:,0]), np.average(sq2.current_frame[:,:,1]), np.average(sq2.current_frame[:,:,2])]
            std2 = np.std(avgs2)
            avg1 = np.average(diff1)
            avg2 = np.average(diff2)
            if avg1 > df['line_clear'] and avg2 > df['line_clear'] and std1 < df['square_diff'] and std2 < df['square_diff']:
                count_white += 1
            else:
                break
    
        if diff1 is not None and diff2 is not None and count_white == 10:
            cleared.append(y)

    return cleared

def square_changed(df_threshold, diff):
    avg = np.average(diff)
    avgs = [np.average(diff[:,:,0]), np.average(diff[:,:,1]), np.average(diff[:,:,2])]
    #print(avgs)
    
    max_c = np.argmax(avgs)

    binary = FrameProcessing.gray_to_binary(diff[:,:,max_c], df_threshold['binarize'])
    area = np.sum(binary)/binary.size

    #print(avg, area)
    
    if avg > df_threshold['piece_diff'] and area > df_threshold['piece_min_area']:
        return True
    else:
        return False


def filled_square(df_threshold, diff):
    avgs = [np.average(diff[:,:,0]), np.average(diff[:,:,1]), np.average(diff[:,:,2])]
    
    max_c = np.argmax(avgs)

    binary = FrameProcessing.gray_to_binary(diff[:,:,max_c], 64)
    
    area = np.sum(binary)/binary.size
    b = np.multiply(diff[:,:,0],binary)
    g = np.multiply(diff[:,:,1],binary)
    r = np.multiply(diff[:,:,2],binary)
    bin_sum = np.sum(binary)
    if bin_sum == 0:
        avgs = [0, 0, 0]
    else:
        avgs = [np.sum(b)/bin_sum, np.sum(g)/bin_sum, np.sum(r)/bin_sum]
    
    std = np.std(avgs)
    
#    print(area, std)
    if area > df_threshold['piece_min_area'] and std > df_threshold['piece_min_std']:
        #print(avg, area, std)
        return True
    else:
        return False

def lock_effect(df_threshold, diff):
    avgs = [np.average(diff[:,:,0]), np.average(diff[:,:,1]), np.average(diff[:,:,2])]
    
    max_c = np.argmax(avgs)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY )
    binary = FrameProcessing.gray_to_binary(diff[:,:,max_c], 64)
    gray = np.multiply(gray, binary)

    area = np.sum(binary)/binary.size
    b = np.multiply(diff[:,:,0],binary)
    g = np.multiply(diff[:,:,1],binary)
    r = np.multiply(diff[:,:,2],binary)
    bin_sum = np.sum(binary)
    if bin_sum == 0:
        avgs= [0, 0, 0]
    else:
        avgs = [np.sum(b)/bin_sum, np.sum(g)/bin_sum, np.sum(r)/bin_sum]
    
    std = np.std(avgs)
    
#    print(area, std)
    if area > df_threshold['piece_min_area'] and std < df_threshold['piece_min_std']:
        #print(avg, area, std)
        return True
    else:
        return False


def piece_on_top(df, playfield, ignore):
    count = 0
    for y in range(df['overlay']['max_row'], playfield.height):
        for x in range(playfield.width):
            if (x, y) not in ignore:
                sq = playfield.frame_board[y][x]
                diff = sq.background_diff()
                if diff is not None:
                    if filled_square(df['threshold'], diff):
                        count += 1
                       
    return count

def playfield_diff(df_threshold, playfield):
    sq_candidates = []
    for y in range(playfield.height):
        for x in range(playfield.width):
            sq = playfield.frame_board[y][x]
            prev_diff = sq.previous_frame_diff()
            if prev_diff is not None:
#                print('SQQQQQQQQQQQ', x, y)
                if square_changed(df_threshold, prev_diff):
                    sq_candidates.append((y,x))
    return sq_candidates
                    
def playfield_frame_diff(df_threshold, cur_playfield, prev_playfield):
    for y in range(playfield.height):
        for x in range(playfield.width):
            cur_sq = playfield.frame_board[y][x]
    

def update_playfield(df, playfield, pieces, ignore):
    for y in range(playfield.height-1):
        for x in range(playfield.width):
            sq = playfield.frame_board[y][x]
            back_diff = sq.background_diff()
            if lock_effect(df['threshold'], back_diff):
                playfield.board[y][x] = df['pieces']['unknown']
            elif filled_square(df['threshold'], back_diff):
                if (x, y) not in ignore:
                    playfield.board[y][x] = FrameProcessing.piece_find_color(pieces, sq.current_frame)
                elif playfield.board[y][x] == df['pieces']['empty']: 
                    playfield.board[y][x] = df['pieces']['unknown']
            else:
                playfield.board[y][x] =  df['pieces']['empty']

def update_board(df, candidate, prev_playfield):
    print('Candidate', candidate)
    temp_playfield = copy.deepcopy(prev_playfield)
        
    piece_id = candidate[0]
    name = df['pieces']['names'][piece_id]
    min_x = candidate[1]
    rotation = candidate[2]
    piece = df['pieces'][name]['rotations'][rotation]
    temp_playfield.hard_drop(piece, min_x, piece_id)
    print(temp_playfield.board)
    board_diff = temp_playfield.board_diff(prev_playfield, df['sprint']['num_pieces'], [])
    
    return board_diff

def count_squares_with_pieces(df, playfield):
    count = 0
    for y in range(playfield.height):
        for x in range(playfield.width):
            sq = playfield.board[y][x]
            if sq > df['pieces']['empty'] and sq <= df['sprint']['num_pieces']:
                count += 1
    return count
        
def advance_time(frame_count, fps, s): #advances s seconds to find the final time faster
    frame_count += s*fps
    return frame_count    

def process_video(piece_tracking = False):
    folder = '/home/alexandre/Downloads/'
    fileName = '210629'
    extension = '.mp4'
    player_name = ''

    session = SessionStats()
    pieces_stats = []

    cap = cv2.VideoCapture(folder + fileName + extension)
    
    cap_width = cap.get(3)
    cap_height = cap.get(4)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')

    defaults = Reader.read_yaml()

    scale = defaults['frame']['scale']
    frame_width = int(defaults['video']['width']*scale)
    frame_height = int(defaults['video']['height']*scale)
    timing = TimingStats(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter('/home/alexandre/Desktop/temp/' + fileName + '.avi', fourcc, timing.fps, (frame_width, frame_height))


    playfield = Playfield(defaults['playfield'], scale)

    hold = None
    nexts = None
    current_piece = None
    current_piece_id = -1
    current_piece_srs_rotation = 0
    current_piece_column = 0
    next_piece = None
    next_piece_id = -1
    prev_hold_id = -1

    prev_hold = None
    prev_frame = None
    prev_playfield = None
    frame = None
    frame_digits = init_frame_digits(defaults['timer'], scale)
    pieces = []
    session.timer = []
    timer_prev_digits = []

    first_hold = False
    timer_initialized = False
    started = False
    digits_read = False
    pieces_read = False
    background_top = False
    training_complete = False
    recording = False
    ended = False

    prev_hold_frame = -1
    entry_frame = -1
    end_count = 0
    prev_squares_on_top = 0
    squares_on_top = 0
    
    piece_lines_cleared = 0
    map_frame_time = {}
    ignore_line_clear = 0

    session.piece_count = -1
    colors_defined = 0

    timing.start_frame = total_frames
    timing.end_frame = total_frames
    timing.fastest_time = 13
    timing.slowest_time_corrected = 50
    timing.slowest_time = 120
    timing.total_pieces = 100
    camera = False
    #camera = True
    #timing.start_frame = 116
    #timing.end_frame = 1475
    #final_time = [0,3,1,7,5]
#    pts1 = np.float32([[113, 90], [1200, 80], [116, 700], [1204, 696]])
#    pts2 = np.float32([[0, 0], [960, 0], [0, 540], [960, 540]])
#    pts1 = np.float32([[53, 144], [1171, 125], [152, 567], [1242, 703]])
#    pts2 = np.float32([[84, 143], [892, 99], [154, 439], [960, 540]])


    replay_overlay = int(timing.fps*defaults['overlay']['replay_text'])

    numbers = []
    ignore_squares = init_ignore_squares(defaults)
    ignore_squares_ingame = {}
    ignore_next = 0
    ignore_hold = 0
    frame_pieces_update = -1
    frame_hold_update = -1
    sq_candidates = []
    board_updated = 0
    last_playfield = None
    prev_candidates = []
    board_id = copy.deepcopy(playfield)
    board_id.height = defaults['layout']['playfield']['num_rows']
    board_id.board = np.zeros((board_id.height, board_id.width))
    board_diff = np.zeros_like(playfield.board)

    while cap.isOpened() and session.frame_count < total_frames:
        ret, original_frame = cap.read()
        session.frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if camera:
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(original_frame, matrix, (960, 540))
        else:
            frame = cv2.resize(original_frame, (frame_width, frame_height)) 
        #original_frame = cv2.resize(original_frame, (frame_width, frame_height))
        original_frame = frame

        set_current_frames(defaults, frame, playfield, hold, nexts, frame_digits)
        next_changed = nexts_changed(defaults['threshold'], nexts)
        entry_delay = np.round(defaults['delay']['entry'] * timing.fps)

        if camera and not started and session.frame_count == timing.start_frame: 
            started = True
            digits_read = True
            playfield.set_background_bottom(frame, defaults['overlay']['max_row'])
            hold = init_hold(defaults['hold'], scale)
            nexts = init_nexts(defaults, scale)
            hold.set_background(frame)
            pieces = init_pieces(defaults)
            for i in range(len(nexts)):
                nexts[i].set_background(frame)
                nexts[i].set_current_frame(frame)
                if set_piece_color(defaults, pieces, nexts[i]):
                    print('Color set')
                    colors_defined += 1
            # cap.set(cv2.CAP_PROP_POS_FRAMES,timing.start_frame + timing.fps)
            # ret, frame = cap.read()
            # frame = cv2.resize(frame, (frame_width, frame_height)) 
            # session.frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # print("  Background top set", session.frame_count)
            # background_top = True
            # playfield.set_background_top(frame, defaults['overlay']['max_row'])
            # cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            # training_complete = True
            
        if not camera and not timer_initialized and timer_is_zero(frame_digits, defaults['threshold']):
            print('  Timer initialized')
            timer_initialized = True
            playfield.set_background_bottom(frame, defaults['overlay']['max_row'])
            

        if (not started and timer_initialized and not timer_digits_match(frame_digits, defaults['threshold'])) and not pieces_read:
            timing.start_frame = session.frame_count - 1
            started = True
            hold = init_hold(defaults['hold'], scale)
            nexts = init_nexts(defaults, scale)
            hold.set_background(frame)
            pieces = init_pieces(defaults)
            for i in range(len(nexts)):
                nexts[i].set_background(frame)
                nexts[i].set_current_frame(frame)
                if set_piece_color(defaults, pieces, nexts[i]):
                    print('Color set')
                    colors_defined += 1
            


        if not background_top and next_changed and digits_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES,session.frame_count - 2)
            ret, frame = cap.read()
            frame = cv2.resize(frame, (frame_width, frame_height)) 
            session.frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print("  Background top set", session.frame_count)
            background_top = True
            playfield.set_background_top(frame, defaults['overlay']['max_row'])
            

        if not pieces_read and next_changed:
            id = len(nexts)-1
            if set_piece_color(defaults, pieces, nexts[id]):
                colors_defined += 1
                print('Colors defined', colors_defined)
            if colors_defined == defaults['sprint']['num_pieces']:
                print('All colors defined')
                pieces_read = True

        if training_complete and session.frame_count >= timing.start_frame and not ended:
            if piece_tracking:
                    
                if session.frame_count == timing.start_frame:
                    next_piece = nexts[0].current_frame
                    next_piece_id = FrameProcessing.map_frame_to_piece(defaults, next_piece, pieces)
                    print('Setting next piece id', next_piece_id)
                if hold_changed(defaults['threshold'], hold, session):
                    if prev_hold is not None:
                        current_piece = prev_hold
                        print('Current id was', current_piece_id, 'hold id was', prev_hold_id)
                        current_piece_id , prev_hold_id = prev_hold_id, current_piece_id
                        current_piece_name = defaults['pieces']['names'][current_piece_id]
                        current_piece_column = defaults['pieces'][current_piece_name]['spawn_column'] #change current_piece update into a function later
                        print('Current id is now', current_piece_id, 'hold id is', prev_hold_id)
                    else:
                        first_hold = True 
                        prev_hold_id = current_piece_id
                        print('First Hold. Hold id is', prev_hold_id)
                if next_changed and not first_hold:
                    current_piece = next_piece
                    current_piece_id = next_piece_id
                    current_piece_name = defaults['pieces']['names'][current_piece_id]

                if session.total_lines == 40:
                    timing.end_frame = session.frame_count - defaults['timer']['end_count']
                    ended = True



                timer_prev_digits = session.timer
                session.timer = match_digits(frame_digits, numbers, defaults['threshold'])
                map_frame_time[session.frame_count] = session.timer

                prev_squares_on_top = squares_on_top
                squares_on_top = piece_on_top(defaults, playfield, ignore_squares)

                temp_playfield = copy.deepcopy(playfield)
                update_playfield(defaults, temp_playfield, pieces, [])

                #print(temp_playfield.board)


            else:
                if session.frame_count == timing.start_frame:
                    next_piece = nexts[0].current_frame
                    next_piece_id = FrameProcessing.map_frame_to_piece(defaults, next_piece, pieces)
                    print('Setting next piece id', next_piece_id)
                timer_prev_digits = session.timer
                if camera:
                    session.timer = Time.frames_to_time(session.frame_count-timing.start_frame, timing.fps)
                else:
                    session.timer = match_digits(frame_digits, numbers, defaults['threshold'])
                map_frame_time[session.frame_count] = session.timer

                prev_squares_on_top = squares_on_top
                squares_on_top = piece_on_top(defaults, playfield, ignore_squares)
                #print('Squares on top:', prev_squares_on_top, squares_on_top)
                #if prev_squares_on_top > 0 and squares_on_top == 0:
                    
                if ignore_line_clear > 0:
                    ignore_line_clear -= 1
                else:
                    temp_lines = line_cleared(defaults['threshold'], playfield)
                    if len(temp_lines) > 0:
                        piece_lines_cleared = len(temp_lines)
                        ignore_line_clear = 20
                        playfield.clear_lines(temp_lines)
                        #board_id.clear_lines(temp_lines)
                        board_updated = session.piece_count+1
                    if len(temp_lines) == 4:
                        min_line = min(temp_lines)
                        board_diff = [(6,0), (6,1), (6,2), (6,3)]
                        board_id.update_with(board_diff, session.piece_count + 1, session.total_lines)        
                        expiry = int(defaults['overlay']['tetris_animation']*timing.fps + session.frame_count)
                        ignore_squares_ingame = update_ignore_tetris(defaults['overlay'], ignore_squares_ingame, playfield.width, playfield.height, min_line, expiry)
                if ignore_next > 0:
                    ignore_next -= 1
                if ignore_hold > 0:
                    ignore_hold -= 1
                if next_changed:
                    entry_delay = np.round(defaults['delay']['entry'] * timing.fps)
                    if ignore_next == 0:
                        ignore_next = entry_delay
                        if not first_hold: #locked piece
                            if session.piece_count >= 0:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, session.frame_count-entry_delay-1)
                                ret, prev_frame = cap.read()
                                if camera:
                                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                                    prev_frame = cv2.warpPerspective(prev_frame, matrix, (960, 540))
                                else:
                                    prev_frame = cv2.resize(prev_frame, (frame_width, frame_height)) 
                                playfield.set_current_frame(prev_frame)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, session.frame_count)

                                temp_playfield = copy.deepcopy(playfield)
                                update_playfield(defaults, temp_playfield, pieces, ignore_squares_ingame)
                                ignore_squares_ingame = remove_from_ignore(ignore_squares_ingame, session.frame_count)
                                if prev_playfield is not None:
                                    temp_playfield.fill_incorrect_squares(prev_playfield)

                                prev_playfield = copy.deepcopy(playfield)

                                print('Session', session.piece_count, board_updated)
                                print(temp_playfield.board)
                                print('Prev', session.piece_count, board_updated)
                                print(prev_playfield.board)

                                if piece_lines_cleared == 0:
                                    board_diff = temp_playfield.board_diff(prev_playfield, defaults['sprint']['num_pieces'], ignore_squares_ingame)
                                    if len(board_diff) != 4:
                                        print('Missing square here!', len(board_diff))
                                        print(prev_playfield.board)
                                        name = defaults['pieces']['names'][current_piece_id]
                                        print(temp_playfield.board)
                                        if len(prev_candidates) == 0:
                                            candidates = temp_playfield.reconstruct_board(defaults['pieces'], prev_playfield, defaults['pieces'][name])
                                        else:
                                            candidates = temp_playfield.reconstruct_board_from_candidates(defaults['pieces'], prev_playfield, prev_candidates, defaults['pieces'][name])
                                        
                                        #Manually fix missing pieces
                                        #print('Count', session.piece_count)
                                        #if session.piece_count == 80:
                                        #    candidates = [[(3, 7, 0)]]
                                        #if session.frame_count == 954:
                                        #    candidates = [[(4, 6, 3)]]

                                        
                                        
                                        
                                        if len(candidates) == 1:
                                            for c in candidates[0]:
                                                print('cand', c)
                                                board_diff = update_board(defaults, c, prev_playfield)
                                                piece = defaults['pieces']['names'][c[0]]
                                                print(piece)
                                                piece_state = np.array(defaults['pieces'][piece]['rotations'][c[2]])
                                                prev_playfield.hard_drop(piece_state, c[1], c[0])
                                                
                                                board_updated += 1
                                                playfield.update_with(board_diff, current_piece_id, 0)
                                                board_id.update_with(board_diff, board_updated, session.total_lines)

                                            prev_candidates = []
                                        else:
                                            print(len(candidates), ' Found', candidates)
                                            board_diff = []
                                            prev_candidates= candidates
                                            last_playfield = prev_playfield
                                    else:
                                        board_updated = session.piece_count + 1
                                        playfield.update_with(board_diff, current_piece_id, 0)
                                        board_id.update_with(board_diff, board_updated, session.total_lines)
                                    #print(board_id.board)
                                            
                                    playfield.remove_effect(defaults)
                                    playfield.set_current_frame(frame)
                                
                            #prev_frame = frame

                            session.piece_count += 1
                            if session.piece_count > 0:
                                print('Next changed', session.frame_count, session.piece_count)
                                hold_frame = -1
                                if prev_hold_frame >= entry_frame:
                                    hold_frame = prev_hold_frame
                                    session.holds += 1
                                    time_diff = Time.frame_to_time_diff(hold_frame, entry_frame, map_frame_time, defaults['timer'])
                                    session.hold_time += time_diff
                                if piece_lines_cleared > 0:
                                    session.update_lines(defaults['delay'], piece_lines_cleared)
                                pc = False
                                if session.perfect_clear():
                                    pc = True
                                    session.perfect_clears += 1
                                piece_stat = PieceStats(session.piece_count, current_piece_id, entry_frame, session.frame_count, hold_frame, piece_lines_cleared, pc, current_piece, hold.current_frame, board_diff)
                                print('Creating ', piece_stat.count, piece_stat.id)
                                pieces_stats.append(piece_stat)
                            entry_frame = session.frame_count
                        else:
                            first_hold = False
                        frame_pieces_update = session.frame_count
                        current_piece = next_piece
                        current_piece_id = next_piece_id
                        print('Current piece is now', current_piece_id)
                
                        piece_lines_cleared = 0
                        if camera:
                            frame_pieces_update += entry_delay
                if session.frame_count == frame_pieces_update: 
                    next_piece = nexts[0].current_frame
                    next_piece_id = FrameProcessing.map_frame_to_piece(defaults, next_piece, pieces)
                    print('Next piece is ', next_piece_id)
                if hold_changed(defaults['threshold'], hold, session):
                    if ignore_hold == 0:
                        ignore_hold = entry_delay
                        print('Hold changed', session.frame_count)
                        if prev_hold is not None:
                            current_piece = prev_hold
                            print('Current id was', current_piece_id, 'hold id was', prev_hold_id)
                            current_piece_id , prev_hold_id = prev_hold_id, current_piece_id
                            print('Current id is now', current_piece_id, 'hold id is', prev_hold_id)
                        else:
                            first_hold = True 
                            prev_hold_id = current_piece_id
                            print('First Hold. Hold id is', prev_hold_id)
                
                        prev_hold_frame = session.frame_count
                        frame_hold_update = session.frame_count
                        if camera:
                            frame_hold_update += entry_delay
                if session.frame_count == frame_hold_update:
                    print('Updating hold frame', session.frame_count)
                    prev_hold = hold.current_frame


                if session.total_lines == 40:
                    timing.end_frame = session.frame_count - defaults['timer']['end_count']
                    ended = True

                # if np.array_equal(timer_prev_digits, session.timer) or session.frame_count >= timing.end_frame:
                #     end_count += 1
                #     if camera:
                #         session.timer = final_time
                #     if end_count >= defaults['timer']['end_count']:
                #         timing.end_frame = session.frame_count - defaults['timer']['end_count']
                #         print("  Final time:", session.timer, timing.end_frame)
                #         ended = True
                # else:
                #     end_count = 0
        if session.frame_count > timing.start_frame and session.frame_count < timing.start_frame + timing.fps and not digits_read:
            frame_count_diff = session.frame_count - timing.start_frame
            div = round(timing.fps)//10
            remainder = div//2
            if frame_count_diff % div == remainder:
                print("Digits read", len(numbers))
                numbers = store_number(numbers, frame_digits[defaults['timer']['decisecond']], defaults['threshold']['digit'])
                if len(numbers) == defaults['timer']['numbers']:
                    #frame_count = advance_time(frame_count,fps,defaults['timer']['time_advance'])
                    digits_read = True                    
                    #cap.set(cv2.CAP_PROP_POS_FRAMES,frame_count)

        if not training_complete and background_top and digits_read and pieces_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            training_complete = True

        back_diff = Draw.draw_background_diff(playfield, hold, frame)
        #if session.frame_count > 1:
        #    prev_diff = Draw.draw_previous_frame_diff(playfield, hold, frame, nexts)
        #current = Draw.draw_current_frame(defaults, playfield, frame_digits, current_piece, next_piece, frame)

        last_piece_time = []
        if session.piece_count > 0:
            entry_delay = int(defaults['delay']['entry'] * timing.fps)
            last_piece_frame = pieces_stats[session.piece_count-1].end_frame - entry_delay
            last_piece_time = map_frame_time[last_piece_frame]

        if session.frame_count < timing.start_frame:
            session.timer = [0,0,0,0,0]
        elif training_complete and session.frame_count < timing.end_frame + 3*timing.fps:
            print(session.frame_count, timing.fps, timing.end_frame + 2*timing.fps)
            original_frame = Draw.draw_time_stats(defaults, original_frame, frame_width, frame_height, session, last_piece_time, board_id, pieces_stats, map_frame_time, timing)
            original_frame = Draw.draw_pieces_stats(defaults, original_frame, pieces_stats, frame_width, frame_height, map_frame_time, timing, session)
            original_frame = Draw.draw_lines_stats(defaults, original_frame, frame_width, frame_height, timing, session, last_piece_time)
            original_frame = Draw.draw_playfield(defaults, original_frame, frame_width, frame_height, board_id, pieces_stats, timing, map_frame_time)
            #original_frame = Draw.draw_player_info(defaults['layout']['player'], original_frame, frame_width, frame_height, player_name)
        
        #original_frame = Draw.draw_histogram(defaults, original_frame, pieces_stats, frame_width, frame_height, map_frame_time, timing, session) 

        print("Frame", session.frame_count, "/", total_frames, session.timer)
        if recording:
            out.write(original_frame)

        if training_complete:
            recording = True

        cv2.imshow('frame',original_frame)
        #if prev_frame is not None:
        #    cv2.imshow('timer', prev_frame)
        #cv2.imshow('back_diff',back_diff)
        #if session.frame_count > 1:
        #    cv2.imshow('prev_diff',prev_diff)

        key = cv2.waitKey(1)
        #while key not in [ord('q'), ord('k'),ord('l')]:
        #    key = cv2.waitKey(0)
        if key == ord('l'):
            session.frame_count -= 5
            if session.frame_count < 0:
                session.frame_count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, session.frame_count)
        elif key == ord('q'):
            break

        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(True)
