import numpy as np
import cv2
from frame_processing import FrameProcessing

class FrameRectangle:
    def __init__(self, min_x, min_y, width, height):
        self.min_x = int(min_x)
        self.max_x = int(min_x + width)
        self.min_y = int(min_y)
        self.max_y = int(min_y + height)
        self.background = None
        self.previous_frame = None
        self.current_frame = None

    def set_background(self, frame):
        self.background = frame[self.min_y:self.max_y, self.min_x:self.max_x]

    def set_current_frame(self, frame):
        self.previous_frame = self.current_frame
        self.current_frame = frame[self.min_y:self.max_y, self.min_x:self.max_x]
        
    def background_diff(self):
        if self.current_frame is not None and self.background is not None:
            max_vals = np.maximum(self.current_frame, self.background)

            return (max_vals-self.background)
        else:
            return None

    def previous_frame_diff(self):
        if self.current_frame is not None and self.previous_frame is not None:
            max_vals = np.maximum(self.current_frame, self.previous_frame)

            return (max_vals-self.previous_frame)
        else:
            return None
    
    def binarize_current_frame(self):
        if self.current_frame is not None:
            return FrameProcessing.binarize(self.current_frame)
        else:
            return None

