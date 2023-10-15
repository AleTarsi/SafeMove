import numpy as np
import cv2

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


def ComputeAngle(base_link_vector,following_link_vector):
    
    '''
    Parameters: base_link_vector,following_link_vector -> 3D vectors representing a human segment
    
    Output: Angle in 
    '''
    cross_product = np.cross(base_link_vector,following_link_vector)
    sin_theta = np.absolute(cross_product)/(np.absolute(base_link_vector)*np.absolute(following_link_vector))
    theta = np.arcsin(sin_theta)
    
    return theta

def ImageCoordinateFrame(image):
    cv2.line(image, np.array([0,0]), np.array([20,0]), (255, 0, 0), 3)
    cv2.line(image, np.array([0,0]), np.array([0,20]), (0, 0, 255), 3)
    
    cv2.putText(image, 'x', (30,25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
    cv2.putText(image, 'y', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)

