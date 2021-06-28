import numpy as np
import cv2

class FrameProcessing:
    @staticmethod
    def gray_to_binary(frame, threshold):
        if frame is not None:
            binary_frame = (frame > threshold).astype(int)
            return binary_frame
        else:
            return None
    
    @staticmethod
    def color_to_binary(frame, threshold):
        if frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
            binary_frame = (gray_frame > threshold).astype(int)
            return binary_frame
        else:
            return None


    @staticmethod
    # Assumes frame_1 and frame_2 are binary images
    def compute_match(frame_1, frame_2):
        if frame_1 is not None and frame_2 is not None:
            frame_match = np.multiply(frame_1, frame_2)
            maximum = np.maximum(frame_1, frame_2)

            score = np.sum(frame_match)/(np.sum(maximum))
            return score
        else:
            return None

    @staticmethod
    def remove_zero_rows_cols(frame):
        s = np.sum(frame, axis = 0)
        to_keep = s > frame.shape[0]/10
        frame = frame[:,to_keep]
        
        to_keep = np.sum(frame, axis = 1) > frame.shape[1]/10
        frame = frame[to_keep,:]

        return frame

    @staticmethod
    def multiply_binary(frame, binary):
        bin2 = binary.copy()
        bin3 = binary.copy()
        
        binary3d = np.stack((binary, bin2, bin3), axis = 2)
        frame = frame * binary3d 
        return frame

    @staticmethod
    def average_color(frame, binary):
        frame = FrameProcessing.multiply_binary(frame, binary)

        div = np.sum(binary)
        avgs =  np.array([int(np.sum(frame[:,:,0])/div), int(np.sum(frame[:,:,1])/ div), int(np.sum(frame[:,:,2])/div)])
        max_val = max(avgs)
        avgs_normalized = avgs/max_val

        return avgs_normalized

    @staticmethod
    def piece_proportion(df_sizes, height, width):
        min_error = 1
        best_fit = 0
        if width > height:
            proportion = width/height
        else:
            proportion = height/width
        for val in df_sizes:
            error = abs(val-proportion)/val
            if error < min_error:
                min_error = error
                best_fit = val
        return best_fit

    @staticmethod
    def resize_piece(binary_piece, proportion):
        if proportion == 1:
            width = 2
            height = 2
        elif proportion == 1.5:
            width = 3
            height = 2
        else:
            width = 4
            height = 1
        resized_sum = np.zeros((height, width))
        square_num_rows = binary_piece.shape[0]/height
        square_num_cols = binary_piece.shape[1]/width
        
        for y in range(binary_piece.shape[0]):
            for x in range(binary_piece.shape[1]):
                r_y = int(y/square_num_rows)
                r_x = int(x/square_num_cols)

                resized_sum[r_y][r_x] += binary_piece[y][x]
        
        resized_sum = resized_sum.flatten()
        order = np.argsort(resized_sum)
        resized = np.zeros((height, width))
        resized.fill(1)

        for i in range(order.size-4):
            val = order[i]
            r_y = val//width
            r_x = val%width
            resized[r_y][r_x] = 0

        #binary_piece = np.array(binary_piece, dtype='uint8')
        #resized = cv2.resize(binary_piece, (width, height)) 
        #print(resized)

        return resized

    @staticmethod
    def match_piece(piece, binary_piece):
        if np.array_equal(piece.standard_position, binary_piece):
            return True
        else:
            return False


    @staticmethod
    def match_binary_piece(pieces, binary_piece, proportion):
        binary_piece = FrameProcessing.resize_piece(binary_piece, proportion)
        print(binary_piece)
        for i in range(len(pieces)):
            piece = pieces[i]
            if FrameProcessing.match_piece(piece, binary_piece):
                print('Matched to', piece.id)
                return piece.id
        return -1

    @staticmethod
    def map_frame_to_piece(df, frame, pieces):
        binary = FrameProcessing.color_to_binary(frame, df['threshold']['piece_intensity'])
        cropped = FrameProcessing.remove_zero_rows_cols(binary)
        proportion = FrameProcessing.piece_proportion(df['pieces']['proportions'], cropped.shape[0],cropped.shape[1] )
        return FrameProcessing.match_binary_piece(pieces, cropped, proportion)


    @staticmethod
    def piece_find_color(pieces, sq):
        avgs = np.array([int(np.average(sq[:,:,0])), int(np.average(sq[:,:,1])), int(np.average(sq[:,:,2])) ])
        max_val = max(avgs)
        avgs = avgs/max_val
        min_error = 3
        best_match = -1
        for i in range(len(pieces)):
            color = pieces[i].color
            diff = np.square(avgs - color)
            error = np.sum(diff)
#            print(pieces[i].id, avgs, color, diff, error)
            
            if error < min_error:
                min_error = error
                best_match = pieces[i].id
        return best_match     