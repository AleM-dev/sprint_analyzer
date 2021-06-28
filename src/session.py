import numpy as np

class SessionStats:
    def __init__(self):
        self.frame_count = 0
        self.piece_count = -1
        self.singles = 0
        self.doubles = 0
        self.triples = 0
        self.tetrises = 0
        self.holds = 0
        self.perfect_clears = 0
        self.line_clears = 0
        self.total_lines = 0
        self.play_time = 0
        self.hold_time = 0
        self.line_clear_time = 0
        self.timer = []

    def update_lines(self, df_delay, lines):
        self.line_clears += 1
        self.total_lines += lines
        if self.perfect_clear():
            lines = df_delay['perfect_clear']
        self.line_clear_time += df_delay['line_clear'][lines]

    def perfect_clear(self):
        if self.total_lines * 10 == self.piece_count * 4:
            return True
        else:
            return False

class TimingStats:
    def __init__(self, fps):
        self.fps = fps
        self.start_frame = -1
        self.end_frame = -1
        self.total_pieces = 0
        self.fastest_time = fps*10000
        self.slowest_time = -1
        self.slowest_time_corrected = -1

    def update_fastest_time(self, val):
        if(self.fastest_time < val):
            self.fastest_time = val

    def update_slowest_time(self, val):
        if(self.slowest_time > val):
            self.slowest_time = val

    def update_slowest_time_corrected(self, val):
        if(self.slowest_time_corrected > val):
            self.slowest_time_corrected = val

class PieceStats:
    def __init__(self, count, id, start_frame, end_frame, hold_frame, lines_cleared, perfect_clear, piece, held_piece, squares):
        self.count = count
        self.id = id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.hold_frame = hold_frame
        if hold_frame > 0:
            self.held = True
        else:
            self.held = False
        self.lines_cleared = lines_cleared
        self.perfect_clear = perfect_clear
        if lines_cleared > 0:
            self.line_cleared = True
        else:
            self.line_cleared = False
        self.piece = piece
        self.held_piece = held_piece
        self.squares = squares
        print("Creating piece", id, start_frame, hold_frame, self.held)
        