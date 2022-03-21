import numpy as np
import cv2
from timer import Time
from PIL import ImageFont, ImageDraw, Image

from frame_processing import FrameProcessing

class Draw:
    @staticmethod
    def draw_playfield_background(playfield, frame):
        frame_background = np.zeros_like(frame)

        for x in range(playfield.width):
            for y in range(playfield.height):
                sq = playfield.frame_board[y][x]
                roi = sq.background
                if roi is not None:
                    frame_background[sq.min_y: sq.max_y, sq.min_x: sq.max_x] = roi

        return frame_background

    @staticmethod
    def draw_current_frame(df, playfield, frame_digits, current_piece, next_piece, frame):
        frame_current = np.zeros_like(frame)

        frame_current = Draw.draw_current_playfield(df, playfield, frame_current)
        frame_current = Draw.draw_current_timer(frame_digits, frame_current)
        frame_current = Draw.draw_current_piece(current_piece, frame_current)
        frame_current = Draw.draw_next_piece(next_piece, frame_current)

        return frame_current

    @staticmethod
    def draw_current_playfield(df, playfield, frame_current):
        for y in range(playfield.height):
            text = str(y)
            text_x = playfield.frame_board[y][0].min_x-40
            text_y = playfield.frame_board[y][0].max_y
            cv2.putText(frame_current, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            for x in range(playfield.width):
                sq = playfield.frame_board[y][x]
                if playfield.board[y][x] > 0:
                    roi = sq.current_frame
                    stds = np.std(roi, axis = 2)
                    stds = FrameProcessing.gray_to_binary(stds, df['threshold']['piece_min_std'])
                    roi = FrameProcessing.multiply_binary(roi, stds)
                    if roi is not None:
                        frame_current[sq.min_y: sq.max_y, sq.min_x: sq.max_x] = roi
        return frame_current

    @staticmethod
    def draw_current_timer(frame_digits, frame_current):
        for i in range(len(frame_digits)):
            rect = frame_digits[i]
            roi = rect.current_frame
            if roi is not None:
                frame_current[rect.min_y: rect.max_y, rect.min_x: rect.max_x] = roi
        return frame_current

    @staticmethod
    def draw_current_piece(current_piece, frame_current):
        if current_piece is not None:
            min_x = 500
            max_x = current_piece.shape[1] + min_x
            min_y = 300
            max_y = current_piece.shape[0] + min_y
            frame_current[min_y: max_y, min_x: max_x] = current_piece
        return frame_current

    @staticmethod
    def draw_next_piece(current_piece, frame_current):
        if current_piece is not None:
            min_x = 550
            max_x = current_piece.shape[1] + min_x
            min_y = 300
            max_y = current_piece.shape[0] + min_y
            frame_current[min_y: max_y, min_x: max_x] = current_piece
        return frame_current

    @staticmethod
    def draw_hold_background_diff(hold, frame, frame_diff):
        diff = hold.background_diff()
        if diff is not None:
            frame_diff[hold.min_y: hold.max_y,hold. min_x: hold.max_x] = diff
        return frame_diff

    @staticmethod
    def draw_playfield_background_diff(playfield, frame, frame_diff):
        for x in range(playfield.width):
            for y in range(playfield.height):
                sq = playfield.frame_board[y][x]
                diff =  sq.background_diff()
                if diff is not None:
                    frame_diff[sq.min_y: sq.max_y, sq.min_x: sq.max_x] = diff
        return frame_diff

    @staticmethod
    def draw_nexts_previous_frame_diff(nexts, frame, frame_diff):
        for i in range(len(nexts)):
            diff = nexts[i].previous_frame_diff()
            if diff is not None:
                frame_diff[nexts[i].min_y: nexts[i].max_y, nexts[i].min_x: nexts[i].max_x] = diff
        

        return frame_diff

    @staticmethod
    def draw_hold_previous_frame_diff(hold, frame, frame_diff):
        diff = hold.previous_frame_diff()
        if diff is not None:
            frame_diff[hold.min_y: hold.max_y,hold. min_x: hold.max_x] = diff
        return frame_diff

    @staticmethod
    def draw_playfield_previous_frame_diff(playfield, frame, frame_diff):
        for x in range(playfield.width):
                for y in range(playfield.height):
                    sq = playfield.frame_board[y][x]
                    diff =  sq.previous_frame_diff()
                    if diff is not None:
                        frame_diff[sq.min_y: sq.max_y, sq.min_x: sq.max_x] = diff
        return frame_diff

    @staticmethod
    def draw_background_diff(playfield, hold, frame):
        frame_diff = np.zeros_like(frame)

        frame_diff = Draw.draw_playfield_background_diff(playfield, frame, frame_diff)
        if hold is not None:
            frame_diff = Draw.draw_hold_background_diff(hold, frame, frame_diff)

        return frame_diff

    @staticmethod
    def draw_previous_frame_diff(playfield, hold, frame, nexts):
        frame_diff = np.zeros_like(frame)

        frame_diff = Draw.draw_playfield_previous_frame_diff(playfield, frame, frame_diff)
        if hold is not None:
            frame_diff = Draw.draw_hold_previous_frame_diff(hold, frame, frame_diff)
        if nexts is not None:
            frame_diff = Draw.draw_nexts_previous_frame_diff(nexts, frame, frame_diff)

        return frame_diff

    @staticmethod
    def draw_player_info(df_layout, frame, width, height, name):
        overlay = frame.copy()
        
        min_x = int(df_layout['min_x']*width)
        min_y = int(df_layout['min_y']*height)
        max_x = int(min_x + df_layout['width']*width)-1
        max_y = int(min_y + df_layout['height']*height)-1
        color = (df_layout['background_color'][0], df_layout['background_color'][1], df_layout['background_color'][2])
        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), color, -1) 

        font_size = height*df_layout['height']/((1+df_layout['font_gap_ratio'])*len(df_layout['text']))

        #draw name
        i = 0
        text = df_layout['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        font_scale = font_size/text_height
        h_gap = int(df_layout['horizontal_gap']*width*df_layout['width'])
        text_x = min_x + h_gap
        v_gap = font_size*df_layout['font_gap_ratio']
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_layout['description_color'][0], df_layout['description_color'][1], df_layout['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        fontpath ='../fonts/k_gothic.ttf' 
        j_font_size = int(font_size * 1.5)
        font = ImageFont.truetype(fontpath, j_font_size)

        img_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(img_pil)

        b = df_layout['value_color'][0]
        g = df_layout['value_color'][1]
        r = df_layout['value_color'][2]
        a = 1
        text_width = draw.textsize(name, font)[0]

        draw.text((text_x, text_y - font_size), name, font = font , fill = (b, g, r, a) ) 

        overlay = np.array(img_pil)         
        
        cv2.addWeighted(overlay,df_layout['alpha'], frame, 1 -df_layout['alpha'], 0, frame)
        return frame

    @staticmethod
    def draw_time_stats(df, frame, width, height, session, last_piece_time, board_id, pieces_stats, map_frame_time, timing):
        overlay = frame.copy()

        df_layout = df['layout']['time']
        df_timer = df['timer']

        min_x = int(df_layout['min_x']*width)
        min_y = int(df_layout['min_y']*height)
        max_x = int(min_x + df_layout['width']*width)-1
        max_y = int(min_y + df_layout['height']*height)-1
        color = (df_layout['background_color'][0], df_layout['background_color'][1], df_layout['background_color'][2])
        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), color, -1) 

        font_size = height*df_layout['height']/((1+df_layout['font_gap_ratio'])*len(df_layout['text']))

        #draw time
        i = 0
        text = df_layout['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        font_scale = font_size/text_height
        h_gap = int(df_layout['horizontal_gap']*width*df_layout['width'])
        text_x = min_x + h_gap
        v_gap = font_size*df_layout['font_gap_ratio']
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_layout['description_color'][0], df_layout['description_color'][1], df_layout['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        centi = Time.time_to_centiseconds(session.timer, df_timer)
        text = Time.centiseconds_to_minutes_string(centi)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - int(df_layout['horizontal_gap']*width*df_layout['width']) - int(text_width)
        text_color = (df_layout['value_color'][0], df_layout['value_color'][1], df_layout['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        cv2.addWeighted(overlay,df_layout['alpha'], frame, 1 -df_layout['alpha'], 0, frame)

        #pps
        i = 1
        text =df_layout['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        font_scale = font_size/text_height
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_layout['description_color'][0],df_layout['description_color'][1],df_layout['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        centi = Time.time_to_centiseconds(last_piece_time, df_timer)
        pps = 0
        if centi > 0:
            pps = 100*session.piece_count/centi
        text = str("%.3f" % pps)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - int(df_layout['horizontal_gap']*width*df_layout['width']) - int(text_width)
        text_color = (df_layout['value_color'][0],df_layout['value_color'][1],df_layout['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        #pps corrected
        i = 2
        text =df_layout['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        font_scale = font_size/text_height
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_layout['description_color'][0],df_layout['description_color'][1],df_layout['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        lcd = session.line_clear_time
        ppsc = 0
        if centi > 0:
            ppsc = session.piece_count/(centi/100-lcd)
        text = str("%.3f" % ppsc)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - int(df_layout['horizontal_gap']*width*df_layout['width']) - int(text_width)
        text_color = (df_layout['value_color'][0],df_layout['value_color'][1],df_layout['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        #estimated time (pc)

        #old esimation
        i = 3
        min_pieces = int(df['sprint']['lines'] * df['playfield']['width']/4)
        remaining_lines = df['sprint']['lines'] - session.total_lines

        text = df_layout['text'][i]

        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        font_scale = font_size/text_height
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_layout['description_color'][0],df_layout['description_color'][1],df_layout['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        estimated_time = 0
        if ppsc > 0:
            estimated_time = Time.time_to_centiseconds(last_piece_time, df_timer)
            if remaining_lines > 0:
                remaining_tetrises = np.ceil(remaining_lines/4)
                pc_index = df['delay']['perfect_clear']
                remaining_pieces = min_pieces - session.piece_count
                estimated_time += 100*(remaining_pieces/ppsc)
                estimated_time += 100*((remaining_tetrises-1)*df['delay']['line_clear'][4] + df['delay']['line_clear'][pc_index])
                text = Time.centiseconds_to_minutes_string(estimated_time)
            else:
                estimated_time = Time.time_to_centiseconds(session.timer, df_timer)
        
        text = Time.centiseconds_to_minutes_string(estimated_time)

        #new estimation
        # entry_delay = int(df['delay']['entry']*timing.fps)

        # squares_per_col = np.zeros(board_id.width)
        # time_per_col = np.zeros(board_id.width)
            
        # for y in range(board_id.height):
        #     for x in range(board_id.width):
        #         sq = int(board_id.board[y][x])
        #         if sq > 0 and sq <= len(pieces_stats):
        #             stat = pieces_stats[sq-1]
        #             entry_frame = stat.start_frame - entry_delay
        #             lock_frame = stat.end_frame - entry_delay
        #             entry_time = map_frame_time[entry_frame]
        #             lock_time = map_frame_time[lock_frame]

        #             if sq == 1:
        #                 diff_corrected = Time.time_to_centiseconds(lock_time, df['timer'])
        #             else:
        #                 diff_corrected = Time.time_diff_without_lcd(lock_time, entry_time, stat.lines_cleared, df)
        #             squares_per_col[x] += 1
        #             time_per_col[x] += diff_corrected
        # print(time_per_col)
                
        # estimated_time = 0
        
        # if np.count_nonzero(squares_per_col == 0) == 0:
        #     avg_per_col = time_per_col/squares_per_col
            
        #     remain_per_col = 40 - squares_per_col
        #     remain_per_col[remain_per_col < 0] = 0
        #     estimated_time = Time.time_to_centiseconds(last_piece_time, df_timer)
        #     if remaining_lines > 0:
        #         avg_time = remain_per_col*avg_per_col
        #         print(avg_per_col)
        #         estimated_time += (np.sum(avg_time)/4)

        #         remaining_tetrises = np.ceil(remaining_lines/4)
        #         pc_index = df['delay']['perfect_clear']
        #         remaining_pieces = min_pieces - session.piece_count
        #         estimated_time += 100*((remaining_tetrises-1)*df['delay']['line_clear'][4] + df['delay']['line_clear'][pc_index])
        #         text = Time.centiseconds_to_minutes_string(estimated_time)
        #     else:
        #         estimated_time = Time.time_to_centiseconds(session.timer, df_timer)
        # else:
        #     text = '?'

        text_color = (df_layout['value_color'][0],df_layout['value_color'][1],df_layout['value_color'][2])
        
        squares_per_col = np.count_nonzero(board_id.board, axis=0)
        squares_over = 0
        if np.max(squares_per_col > df['sprint']['lines']):
            text_color = (0, 0, 255)
            squares_col_over = squares_per_col - df['sprint']['lines']
            squares_col_over[squares_col_over < 0] = 0
            squares_over = np.sum(squares_col_over)
        if session.piece_count > 100:
            text = '-'



        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - int(df_layout['horizontal_gap']*width*df_layout['width']) - int(text_width)
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)


        #estimated time (101+)

        i = 4
        min_pieces = int(df['sprint']['lines'] * df['playfield']['width']/4)
        remaining_lines = df['sprint']['lines'] - session.total_lines


        num_pieces = int(max(101, 100 + np.ceil(squares_over/4)))
        text = df_layout['text'][i][0] + str(num_pieces) + df_layout['text'][i][1]

        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        font_scale = font_size/text_height
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_layout['description_color'][0],df_layout['description_color'][1],df_layout['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        #old estimation
        new_estimated_time = estimated_time
        estimated_time = 0
        if ppsc > 0:
            estimated_time = Time.time_to_centiseconds(last_piece_time, df_timer)
            if remaining_lines > 0:
                remaining_tetrises = np.ceil(remaining_lines/4)
                remaining_pieces = num_pieces - session.piece_count
                estimated_time += 100*(remaining_pieces/ppsc)
                if remaining_lines < 4:
                    estimated_time += 100*df['delay']['line_clear'][remaining_lines]
                else:  
                    estimated_time += 100*((remaining_tetrises)*df['delay']['line_clear'][4])
            else:
                estimated_time = Time.time_to_centiseconds(session.timer, df_timer)
        text = Time.centiseconds_to_minutes_string(estimated_time)
        if session.piece_count == 100 and session.total_lines == 40:
            text = '-'

        #new estimation
        # if new_estimated_time > 0:
        #     estimated_time = new_estimated_time
        #     remaining_squares = 4*(num_pieces - 100) - squares_over
        #     estimated_time += (remaining_squares*100/(4*ppsc))
        #     estimated_time = int(estimated_time)
        #     if remaining_lines > 0:
        #         pc_index = df['delay']['perfect_clear']
        #         estimated_time -= 100*df['delay']['line_clear'][pc_index]
        #         if remaining_lines < 4:
        #             estimated_time += 100*df['delay']['line_clear'][remaining_lines]
        #         else:  
        #             estimated_time += 100*(df['delay']['line_clear'][4])

        #     text = Time.centiseconds_to_minutes_string(estimated_time)
        # else:
        #     text = '?'
        # if remaining_lines == 0 and session.piece_count == min_pieces:
        #     text = '-:--.--'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - int(df_layout['horizontal_gap']*width*df_layout['width']) - int(text_width)
        text_color = (df_layout['value_color'][0],df_layout['value_color'][1],df_layout['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
            

        cv2.addWeighted(overlay,df_layout['alpha'], frame, 1 -df_layout['alpha'], 0, frame)

        return frame

    @staticmethod
    def draw_piece_stats(defaults, overlay, piece):
        return overlay

    @staticmethod
    def bar_color(time, min, max):
        r = int(255 * (time-min)/(max-min))
        b = int(255 * (max-time)/(max-min))
        g = 0
        return (b, g, r)

    @staticmethod
    def draw_pieces_stats(defaults, frame, pieces_stats, width, height, map_frame_time, timing, session):
        overlay = frame.copy()

        defaults_pieces = defaults['layout']['pieces']

        min_x = int(defaults_pieces['min_x']*width)
        min_y = int(defaults_pieces['min_y']*height)
        max_x = int(min_x + defaults_pieces['width']*width)-1
        max_y = int(min_y + defaults_pieces['height']*height)-1
#        print(min_x, min_y, max_x, max_y)
        color = (defaults_pieces['background_color'][0], defaults_pieces['background_color'][1], defaults_pieces['background_color'][2])
        font_size = height*defaults_pieces['height']/((1+defaults_pieces['font_gap_ratio'])*defaults_pieces['amount'])
        #print(font_size)

        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), color, -1) 

        max_i = len(pieces_stats)
        min_i = max_i - defaults_pieces['amount'] + 1
        if min_i < 0:
            min_i = 0 
        for i in range(max_i, min_i, -1):
            stat = pieces_stats[i-1]
            
            #print id
            text = str(stat.count)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            font_scale = font_size/text_height
            h_gap = int(defaults_pieces['horizontal_gap']*width*defaults_pieces['width'])
            text_x = min_x + h_gap
            v_gap = font_size*defaults_pieces['font_gap_ratio']
            text_y = min_y + int(font_scale*text_height -  0.5*v_gap + (i - min_i)*(font_size + v_gap))
            text_color = (defaults_pieces['value_color'][0], defaults_pieces['value_color'][1], defaults_pieces['value_color'][2])
            cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

            #draw hold
            roi = stat.held_piece
            roi_h = int(font_size)
            ratio = font_size/roi.shape[0]
            roi_w = int(roi.shape[1] * ratio)
            roi = cv2.resize(roi, (roi_w, roi_h)) 
            
            roi_x = text_x + int(defaults_pieces['count']*width*defaults_pieces['width'])
            if roi is not None and stat.held:
                overlay[text_y - roi_h: text_y, roi_x: roi_x + roi_w] = roi
                arrow_x = roi_x + int(defaults_pieces['hold']*width*defaults_pieces['width'])
                arrow_x_end = arrow_x + int(defaults_pieces['arrow']*width*defaults_pieces['width']*0.75)
                image = cv2.arrowedLine(overlay, (arrow_x, text_y - int(roi_h*0.25)), (arrow_x_end, text_y - int(roi_h*0.25)),  text_color, 1)
                image = cv2.arrowedLine(overlay, (arrow_x_end, text_y - int(roi_h*0.75)), (arrow_x, text_y - int(roi_h * 0.75)),  text_color, 1)

            #draw piece
            roi = stat.piece
            roi_h = int(font_size)
            ratio = font_size/roi.shape[0]
            roi_w = int(roi.shape[1] * ratio)
            roi = cv2.resize(roi, (roi_w, roi_h)) 
            
            roi_x = roi_x + int((defaults_pieces['hold'] + defaults_pieces['arrow'])*width*defaults_pieces['width'])
            if roi is not None:
                overlay[text_y - roi_h: text_y, roi_x: roi_x + roi_w] = roi
            
            #draw piece time
            entry_delay = int(defaults['delay']['entry']*timing.fps)
            entry_frame = stat.start_frame - entry_delay
            lock_frame = stat.end_frame - entry_delay

            if entry_frame in map_frame_time:
                entry_time = map_frame_time[entry_frame]
            else:
                entry_time=[0,0,0,0,0]
            lock_time = map_frame_time[lock_frame]

            if stat.count == timing.total_pieces:
                lock_time = session.timer
            lines_cleared = stat.lines_cleared
            if stat.perfect_clear:
                lines_cleared = 5
            if stat.count == 1:
                diff = Time.time_to_centiseconds(lock_time, defaults['timer'])
            else:
                diff = Time.time_diff(lock_time, entry_time, defaults['timer'])
            diff_str = Time.centiseconds_to_seconds_string(diff)

            text_x = roi_x + int(defaults_pieces['current']*width*defaults_pieces['width'])
            cv2.putText(overlay, diff_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

            #draw bar
            bar_x = text_x + int(defaults_pieces['piece_time']*width*defaults_pieces['width'])
            bar_max_length = int(defaults_pieces['bar']*width*defaults_pieces['width'])
            ratio_diff = diff / (timing.slowest_time - timing.fastest_time)
            bar_length = int(bar_max_length*ratio_diff)
            
            bar_y = text_y - int(font_size*0.75)
            bar_height = int(font_size*0.5)

            if stat.lines_cleared > 0:
                color = (defaults_pieces['line_clear_color'][0], defaults_pieces['line_clear_color'][1], defaults_pieces['line_clear_color'][2])
                cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), color, -1) 

            diff_corrected = Time.time_diff_without_lcd(lock_time, entry_time, lines_cleared, defaults)
               
            ratio_corrected= (diff_corrected) / (timing.slowest_time - timing.fastest_time)
            bar_length = int(bar_max_length*ratio_corrected)

            color = Draw.bar_color(diff_corrected, timing.fastest_time, timing.slowest_time_corrected)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), color, -1) 

            if stat.held:
                hold_time = map_frame_time[stat.hold_frame - entry_delay]
                diff_hold = Time.time_diff(hold_time, entry_time, defaults['timer'])
                #print('Hold', diff_hold, hold_time, entry_time)

                ratio_hold = (diff_hold) / (timing.slowest_time - timing.fastest_time)
                bar_length = int(bar_max_length*ratio_hold)

                color = (defaults_pieces['hold_color'][0], defaults_pieces['hold_color'][1], defaults_pieces['hold_color'][2])

                cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), color, -1) 


            #draw total time
            time_centi = int(Time.time_to_centiseconds(lock_time, defaults['timer']))
            time_str = Time.centiseconds_to_minutes_string(time_centi)
            
            (text_width, text_height), baseline = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            text_width *= font_scale
            text_x = max_x - h_gap - int(text_width)

            cv2.putText(overlay, time_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

            

        cv2.addWeighted(overlay, defaults_pieces['alpha'], frame, 1 - defaults_pieces['alpha'], 0, frame)

        return frame

    @staticmethod    
    def draw_lines_stats(df, frame, width, height, timing, session, last_piece_time):
        overlay = frame.copy()

        df_lines = df['layout']['lines']

        min_x = int(df_lines['min_x']*width)
        min_y = int(df_lines['min_y']*height)
        max_x = int(min_x + df_lines['width']*width)-1
        max_y = int(min_y + df_lines['height']*height)-1
        h_gap = int(df_lines['horizontal_gap']*width*df_lines['width'])
            
        color = (df_lines['background_color'][0], df_lines['background_color'][1], df_lines['background_color'][2])
        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), color, -1) 

        font_size = height*df_lines['height']/((1+df_lines['font_gap_ratio'])*len(df_lines['text']))
            
        #draw holds
        i = 1
        text = df_lines['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        font_scale = font_size/text_height
        v_gap = font_size*df_lines['font_gap_ratio']
        
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_lines['description_color'][0], df_lines['description_color'][1], df_lines['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        text = str(session.holds)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x += int(df_lines['description_width']*width*df_lines['width']) - int(text_width)
        text_color = (df_lines['value_color'][0], df_lines['value_color'][1], df_lines['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        text = Time.centiseconds_to_seconds_string(session.hold_time) + 's'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - h_gap - int(text_width)
        text_color = (df_lines['value_color'][0], df_lines['value_color'][1], df_lines['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        #draw line clears
        i = 2
        text = df_lines['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_lines['description_color'][0], df_lines['description_color'][1], df_lines['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        text = str(session.line_clears)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x += int(df_lines['description_width']*width*df_lines['width']) - int(text_width)
        text_color = (df_lines['value_color'][0], df_lines['value_color'][1], df_lines['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        line_clear_time = int(100*session.line_clear_time)
        text = Time.centiseconds_to_seconds_string(line_clear_time) + 's'
        
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - h_gap - int(text_width)
        text_color = (df_lines['value_color'][0], df_lines['value_color'][1], df_lines['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        #perfect clears
        i = 3
        text = df_lines['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_lines['description_color'][0], df_lines['description_color'][1], df_lines['description_color'][2])
            
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        text = str(session.perfect_clears)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x += int(df_lines['description_width']*width*df_lines['width']) - int(text_width)
        text_color = (df_lines['value_color'][0], df_lines['value_color'][1], df_lines['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)


        #cleared lines
        i = 4
        text = df_lines['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_lines['description_color'][0], df_lines['description_color'][1], df_lines['description_color'][2])
            
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        text = str(session.total_lines)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x += int(df_lines['description_width']*width*df_lines['width']) - int(text_width)
        text_color = (df_lines['value_color'][0], df_lines['value_color'][1], df_lines['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        #draw pieces
        i = 0
        text = df_lines['text'][i]
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_x = min_x + h_gap
        text_y = min_y + int(font_scale*text_height +  0.5*v_gap + i*(font_size + v_gap))
        text_color = (df_lines['description_color'][0], df_lines['description_color'][1], df_lines['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        count = session.piece_count
        if count < 0:
            count = 0
        text = str(count)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x += int(df_lines['description_width']*width*df_lines['width']) - int(text_width)
        text_color = (df_lines['value_color'][0], df_lines['value_color'][1], df_lines['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        if count == timing.total_pieces:
            last_piece_time = session.timer
            
        piece_time = Time.time_to_centiseconds(last_piece_time, df['timer']) - 100*session.line_clear_time - session.hold_time
#        print('Times', piece_time, last_piece_time, session.line_clear_time, session.hold_time)
        text = Time.centiseconds_to_seconds_string(piece_time) + 's'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - h_gap - int(text_width)
        text_color = (df_lines['value_color'][0], df_lines['value_color'][1], df_lines['value_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)


        cv2.addWeighted(overlay, df_lines['alpha'], frame, 1 - df_lines['alpha'], 0, frame)

        return frame

    @staticmethod
    def draw_playfield(df, frame, width, height, playfield, pieces_stats, timing, map_frame_time):
        overlay = frame.copy()
        df_layout_play = df['layout']['playfield']
        df_pieces = df['pieces']

        min_x = int(df_layout_play['min_x']*width)
        min_y = int(df_layout_play['min_y']*height)
        max_x = int(min_x + df_layout_play['width']*width)-1
        max_y = int(min_y + df_layout_play['height']*height)-1
        h_gap = int(df_layout_play['horizontal_gap']*width*df_layout_play['width'])
        v_gap = int(df_layout_play['vertical_gap']*height*df_layout_play['height'])

        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), df_layout_play['background_color'], -1) 

        #draw text
        text = df_layout_play['text']
        font_size = int((max_y-min_y)/50)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        font_scale = font_size/text_height
        text_x = min_x + int((max_x - min_x - text_width*font_scale)/2)
        text_y = int(min_y + v_gap + text_height*font_scale)
        #print(text_x, text_y, text_width*font_scale)
        text_color = (df_layout_play['description_color'][0], df_layout_play['description_color'][1], df_layout_play['description_color'][2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        
        rect_width = (max_x - min_x - 2*h_gap)/playfield.width
        rect_height = (max_y - min_y - 2*v_gap)/df_layout_play['num_rows']


        entry_delay = int(df['delay']['entry']*timing.fps)
            

        #print(rect_width, rect_height, min_x, min_y, max_x, max_y)

        #draw color scale
        cell_width = (max_x - min_x - 2*h_gap)/(timing.slowest_time_corrected - timing.fastest_time + 1)
        #print(cell_width)
        cell_y = text_y + font_size
        cell_height = font_size/2

        for i in range(timing.fastest_time, timing.slowest_time_corrected + 1):
            color = Draw.bar_color(i, timing.fastest_time, timing.slowest_time_corrected)
            cell_x = min_x + round(min_y + h_gap + (i-timing.fastest_time)*cell_width)
            cv2.rectangle(overlay, (cell_x, cell_y), (round(cell_x + cell_width), round(cell_y + cell_height)), color, -1) 

        #draw fastest/slowest time
        text = str(timing.fastest_time/100) + 's'
        text_x = min_x + h_gap
        text_y = round(cell_y + 2*font_size)
        text_color = Draw.bar_color(timing.fastest_time, timing.fastest_time, timing.slowest_time_corrected)
        text_color = (text_color[0], text_color[1] + 160, text_color[2] + 96)
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        text = str(timing.slowest_time_corrected/100) + 's'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        text_width *= font_scale
        text_x = max_x - h_gap - int(text_width)
        text_color = Draw.bar_color(timing.slowest_time_corrected, timing.fastest_time, timing.slowest_time_corrected)
        text_color = (text_color[0] + 96, text_color[1] + 96, text_color[2])
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)


        #draw line colors
        color = (df_layout_play['line_color'][0], df_layout_play['line_color'][1], df_layout_play['line_color'][2])
        for i in range(1,10):
            text = str(4*i)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            text_width *= font_scale
            text_height *= font_scale
        
            line_y = int(min_y + (playfield.height - (i*4-1) - 1)*rect_height + v_gap)
            line_min_x = min_x
            line_max_x = round(max_x - 2*h_gap - text_width)

            cv2.rectangle(overlay, (line_min_x, line_y), (line_max_x, line_y), color, -1) 
            text_x = line_max_x + h_gap
            text_y = round(line_y + text_height/2)
            cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)





        #draw 40th row line
        y_40 = int(min_y + (playfield.height - 39 - 1)*rect_height + v_gap)
        color = (df_layout_play['row_40_color'][0], df_layout_play['row_40_color'][1], df_layout_play['row_40_color'][2])
        cv2.rectangle(overlay, (min_x, y_40), (int(max_x + rect_width - 1), y_40), color, -1) 

        #draw pieces
        for y in range(playfield.height):
            for x in range(playfield.width):
                sq = int(playfield.board[y][x])
                if sq > 0 and sq <= len(pieces_stats):
                    stat = pieces_stats[sq-1]
                    entry_frame = stat.start_frame - entry_delay
                    lock_frame = stat.end_frame - entry_delay
                    if entry_frame in map_frame_time:
                        entry_time = map_frame_time[entry_frame]
                    else:
                        entry_time=[0,0,0,0,0]
                    lock_time = map_frame_time[lock_frame]

                    diff_corrected = Time.time_diff_without_lcd(lock_time, entry_time, stat.lines_cleared, df)
                    #if stat.lines_cleared > 0:
                    #    print(stat.id, diff_corrected, lock_time, entry_time)
                    color = Draw.bar_color(diff_corrected, timing.fastest_time, timing.slowest_time_corrected)   
        
                    rect_x = int(min_x + x*rect_width + h_gap)
                    rect_y = int(min_y + (playfield.height - y - 1)*rect_height + v_gap)

                    #color = (df_pieces['colors'][sq][0], df_pieces['colors'][sq][1], df_pieces['colors'][sq][2])
                    cv2.rectangle(overlay, (rect_x, rect_y), (int(rect_x + rect_width - 1), int(rect_y + rect_height - 1)), color, -1) 

                    if stat.lines_cleared > 0:
                        if stat.perfect_clear:
                            color = (df_layout_play['perfect_clear_color'][0], df_layout_play['perfect_clear_color'][1], df_layout_play['perfect_clear_color'][2])
                        else:
                            color = (df_layout_play['line_clear_color'][0], df_layout_play['line_clear_color'][1], df_layout_play['line_clear_color'][2])
                        
                    cv2.line(overlay, (rect_x + 3, round(rect_y + rect_height) - 3), (round(rect_x + rect_width - 3), rect_y + 3), color, 1)

                    offset_x = [-1, 1]
                    offset_y = [-1, 1]

                    for dx in offset_x:
                        if x + dx == -1 or x + dx == playfield.width or sq != playfield.board[y][x + dx]:
                            color = (df_layout_play['piece_border_color'][0], df_layout_play['piece_border_color'][1], df_layout_play['piece_border_color'][2])
                            if dx == -1:
                                cv2.rectangle(overlay, (rect_x, rect_y), (rect_x, int(rect_y + rect_height - 1)) , color, -1) 
                            else:
                                cv2.rectangle(overlay, (int(rect_x + rect_width), rect_y), (int(rect_x + rect_width), int(rect_y + rect_height)) , color, -1) 
                                
                    for dy in offset_y:
                        if y + dy == -1 or y + dy == playfield.height or sq != playfield.board[y + dy][x]:
                            color = (df_layout_play['piece_border_color'][0], df_layout_play['piece_border_color'][1], df_layout_play['piece_border_color'][2])
                            if dy == -1:
                                cv2.rectangle(overlay, (rect_x, int(rect_y + rect_height)), (int(rect_x + rect_width), int(rect_y + rect_height)) , color, -1) 
                            else:
                                cv2.rectangle(overlay, (rect_x, rect_y), (int(rect_x + rect_width), rect_y) , color, -1) 
        
        for y in range(playfield.height):
            for x in range(playfield.width):
                sq = int(playfield.board[y][x])
                if sq > 0 and sq <= len(pieces_stats):

                    ignore = False
                    #draw perfect clear border
#                    if pieces_stats[sq-1].lines_cleared > 0 and pieces_stats[sq-1].perfect_clear:
#                        color = (df_layout_play['perfect_clear_color'][0], df_layout_play['perfect_clear_color'][1], df_layout_play['perfect_clear_color'][2])

                    #draw hold border
                    #el
                    if pieces_stats[sq-1].held:
                        color = (df_layout_play['hold_color'][0], df_layout_play['hold_color'][1], df_layout_play['hold_color'][2])

                    else:
                        ignore = True

                    if not ignore:
                        offset_x = [-1, 1]
                        offset_y = [-1, 1]

                        
                        rect_x = int(min_x + x*rect_width + h_gap)
                        rect_y = int(min_y + (playfield.height - y - 1)*rect_height + v_gap)

                        for dx in offset_x:
                            if x + dx == -1 or x + dx == playfield.width or sq != playfield.board[y][x + dx]:
                                if dx == -1:
                                    cv2.rectangle(overlay, (rect_x, rect_y), (rect_x+1, int(rect_y + rect_height - 1)) , color, -1) 
                                else:
                                    cv2.rectangle(overlay, (int(rect_x + rect_width - 1), rect_y), (int(rect_x + rect_width), int(rect_y + rect_height)) , color, -1) 
                                    
                        for dy in offset_y:
                            if y + dy == -1 or y + dy == playfield.height or sq != playfield.board[y + dy][x]:
                                if dy == -1:
                                    cv2.rectangle(overlay, (rect_x, int(rect_y + rect_height)), (int(rect_x + rect_width), int(rect_y + rect_height- 1)) , color, -1) 
                                else:
                                    cv2.rectangle(overlay, (rect_x, rect_y), (int(rect_x + rect_width), rect_y + 1) , color, -1) 
        


        cv2.addWeighted(overlay, df_layout_play['alpha'], frame, 1 - df_layout_play['alpha'], 0, frame)

        return frame

    @staticmethod
    def draw_histogram(df, frame, pieces_stats, width, height, map_frame_time, timing, session):
        overlay = frame.copy()

        df_hist = df['layout']['histogram']

        min_x = int(df_hist['min_x']*width)
        min_y = int(df_hist['min_y']*height)
        max_x = int(min_x + df_hist['width']*width)-1
        max_y = int(min_y + df_hist['height']*height)-1
        wall_width = int((max_x - min_x)*df_hist['wall_width'])
        cell_width = np.round(wall_width / df['playfield']['width'])
        max_min_diff = timing.slowest_time_corrected - timing.fastest_time
        cell_height = np.round((max_y-min_y) / max_min_diff)
        
        
            
#        print(min_x, min_y, max_x, max_y)
        color = (df_hist['background_color'][0], df_hist['background_color'][1], df_hist['background_color'][2])
        #print(font_size)

        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), color, -1) 

        for i in range(len(pieces_stats)):
            stat = pieces_stats[i-1]
            print('Id', i, stat.id)
            
            #draw piece time
            entry_delay = int(df['delay']['entry']*timing.fps)
            entry_frame = stat.start_frame - entry_delay
            lock_frame = stat.end_frame - entry_delay
            entry_time = map_frame_time[entry_frame]
            lock_time = map_frame_time[lock_frame]

            if stat.count == timing.total_pieces:
                lock_time = session.timer
            lines_cleared = stat.lines_cleared
            if stat.perfect_clear:
                lines_cleared = 5

            diff_corrected = Time.time_diff_without_lcd(lock_time, entry_time, lines_cleared, df)
            diff_corrected -= timing.fastest_time

            board_squares = -1
            if len(stat.squares) > 0:
                board_min_x = min(stat.squares)[0]
                board_max_x = max(stat.squares)[0]

                if board_min_x <= df['playfield']['width'] - board_max_x - 1:
                    board_column = board_min_x
                else:
                    board_column = board_max_x

                #r = np.random.randint(0, 5) - 2
                circle_x = int(min_x + board_column*cell_width)
                circle_y = int(min_y + diff_corrected*cell_height)

                #print(circle_x, circle_y, diff_corrected)

                #print('Circle', circle_x, circle_y)

                color = (df['pieces']['colors'][stat.id][0], df['pieces']['colors'][stat.id][1], df['pieces']['colors'][stat.id][2])
                cv2.circle(overlay, (circle_x, circle_y), 8, color, -1, 8, 0)

        cv2.addWeighted(overlay, df_hist['alpha'], frame, 1 - df_hist['alpha'], 0, frame)

        return frame    
