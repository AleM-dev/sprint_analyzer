class Piece:
    def __init__(self, id, standard_position):
        self.id = id
        self.standard_position = standard_position
        self.color_defined = False
        self.color = []
        self.frame = None
    