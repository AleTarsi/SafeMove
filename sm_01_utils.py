import numpy as np


def computeMidPosition(A,B, Lambda = 0.5):
    
    '''
    Parameters: A,B -> 3D vectors
                Lambda -> bilinear interpolation coefficient: 0.5 = half way
    
    Output: Middle point in 3D Euclidean Space.
    '''
    
    A = np.array(A) # Left Hip
    B = np.array(B) # Right Hip

    mid = Lambda*A + (1-Lambda)*B
    return mid 

def fromLandMarkTo3dPose(human_part,landmark,width,height):
    
    '''
    Parameters: A,B -> 3D vectors
                Lambda -> bilinear interpolation coefficient: 0.5 = half way
    
    Output: Middle point in 3D Euclidean Space.
    '''
    
    position = [landmark[human_part].x * width,landmark[human_part].y * height,landmark[human_part].z *width]
    return position

def null():
    return 0
    
