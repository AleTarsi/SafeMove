
import numpy as np
def camera_calibration(img_height, img_width):
    # Camera focal lenght
    focal_length = 1 * img_width

    # The camera matrix
    cam_matrix = np.array([ [focal_length, 0, img_height / 2],
                            [0, focal_length, img_width / 2],
                            [0, 0, 1]])

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    return focal_length, cam_matrix, dist_matrix
    