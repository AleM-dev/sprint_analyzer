class Piece:
    def __init__(self, id, standard_position, dfs = None, max_col = 10):
        self.id = id
        self.standard_position = standard_position
        self.color_defined = False
        self.color = []
        self.frame = None
    
        self.column = dfs['spawn_column']
        self.row = dfs['spawn_row']
        self.max_col = max_col
        self.rotation = 0
        self.srs_rotations = dfs['srs_rotations']

    def min_column():
        

    def max_column():

    def move_left():
        self.column -= 1

    def move_right():
        self.column += 1

    def move_down():
        self.row -= 1

    def rotate_clockwise():
        self.rotation = (self.rotation + 1)%4
        while self.min_column() < 0:
            self.move_right()
        while self.max_column() > max_col:
            self.move_left()
        
    def rotate_anticlockwise():
        self.rotation = (self.rotation + 3)%4
        while self.min_column() < 0:
            self.move_right()
        while self.max_column() > max_col:
            self.move_left()
        
