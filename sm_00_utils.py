import numpy as np
import cv2
 
red = (0,0,255)
green = (0,255,0)
blue = (0,255,0)

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


def Compute3DBaseGivenV1(Vx):
    
    '''
    Parameters: Vx -> x-axis of the new frame
    
    Output: Rotation matrix world to nose, and single vector of this matrix
    '''
    assert Vx.shape == [3,1]
    # take the vector vx = nose vector
    # rotate v_tmp in positive z direction of the original frame
    theta = np.deg2rad(10)
    Vtmp = np.dot(ZRotationMatrix(theta), np.array((Vx)))
    
    if abs(np.dot(Vtmp,Vx)) == 1:
        print("The coordinate Frame are not well defined, due to abs(np.dot(Vtmp,Vx)) == 1")
        
    # take the cross product btw vx and vy and define it as vz
    Vz = np.cross(Vx,Vtmp)
    # take the corss product btw vx and vz and define it as vy
    Vy = - np.cross(Vx,Vz)
    
    Rmat = np.concatenate((Vx,Vy,Vz), axis=1) # Rotation matrix
    
    print (Rmat)
    return Rmat, Vx,Vy,Vz

def ZRotationMatrix(theta):
    sint = np.sin(theta) # sin theta
    cost = np.cos(theta) # cos theta
    return np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])


def ComputeAngle(base_link_vector,following_link_vector):
    
    '''
    Parameters: base_link_vector,following_link_vector -> 3D vectors representing a human segment
    
    Output: Angle in btw the two vectors
    '''
    cross_product = np.cross(base_link_vector,following_link_vector)
    sin_theta = np.absolute(cross_product)/(np.absolute(base_link_vector)*np.absolute(following_link_vector))
    theta = np.arcsin(sin_theta)
    
    return theta

def ImageCoordinateFrame(image):
    cv2.line(image, np.array([0,0]), np.array([20,0]), (255, 0, 0), 3)
    cv2.line(image, np.array([0,0]), np.array([0,20]), (0, 0, 255), 3)
    
    cv2.putText(image, 'x', (30,25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, red, 2)
    cv2.putText(image, 'y', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, green, 2)

def _3DCoordinateFrame(image, rot_vec, trans_vec, cam_matrix, dist_matrix):
    zero3D = np.zeros((3,1))
    
    VbaseX, _ = cv2.projectPoints((3000, 0, 0), zero3D, zero3D, cam_matrix, dist_matrix)
    VbaseY, _ = cv2.projectPoints((0, 3000, 0), zero3D, zero3D, cam_matrix, dist_matrix)
    VbaseZ, _ = cv2.projectPoints((0, 0, 3000), zero3D, zero3D, cam_matrix, dist_matrix)
    
    zero2D = np.zeros((2,), dtype=int)
    cv2.line(image, zero2D, np.array([VbaseX[0][0][0] , VbaseX[0][0][1]], dtype=int), red, 3)
    cv2.line(image, zero2D, np.array([VbaseY[0][0][0] , VbaseY[0][0][1]], dtype=int), green, 3)
    cv2.line(image, zero2D, np.array([VbaseZ[0][0][0] , VbaseZ[0][0][1]], dtype=int), blue, 3)
