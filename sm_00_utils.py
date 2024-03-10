import numpy as np
import cv2
 
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    
class PoseLandmark():
  """The 33 pose landmarks."""
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32
  
def computeFPS(end,start,frames_to_skip):
    totalTime = end - start

    try:
        fps_output = (1 / totalTime)*frames_to_skip
    except:
        fps_output= -1
                
    return fps_output
  
def fromWorldLandmark2nparray(worldLandMark):
    return np.array([worldLandMark.x, worldLandMark.z, worldLandMark.y*(-1)])
  
def computeMidPosition(A,B, Lambda = 0.5):
    
    '''
    Parameters: A,B -> 3D vectors
                Lambda -> bilinear interpolation coefficient: 0.5 = half way
    
    Output: Middle point in 3D Euclidean Space.
    '''
    
    A = np.array(A) # Left Hip
    B = np.array(B) # Right Hip

    mid = Lambda*A + (1-Lambda)*B
    return mid, mid[0], mid[1], mid[2]

def fromLandMarkTo3dPose(human_part,landmark,width,height):
    
    '''
    Parameters: A,B -> 3D vectors
                Lambda -> bilinear interpolation coefficient: 0.5 = half way
    
    Output: Middle point in 3D Euclidean Space.
    '''
    
    position = [landmark[human_part].x * width,landmark[human_part].y * height,landmark[human_part].z *width]
    return position


# def Compute3DBaseGivenV1(Vx):
    
#     '''
#     Parameters: Vx -> x-axis of the new frame
    
#     Output: Rotation matrix world to nose, and single vector of this matrix
#     '''
#     assert Vx.shape == [3,1]
#     # take the vector vx = nose vector
#     # rotate v_tmp in positive z direction of the original frame
#     theta = np.deg2rad(10)
#     Vtmp = np.dot(ZRotationMatrix(theta), np.array((Vx)))
    
#     if abs(np.dot(Vtmp,Vx)) == 1:
#         print("The coordinate Frame are not well defined, due to abs(np.dot(Vtmp,Vx)) == 1")
        
#     # take the cross product btw vx and vy and define it as vz
#     Vz = np.cross(Vx,Vtmp)
#     # take the corss product btw vx and vz and define it as vy
#     Vy = - np.cross(Vx,Vz)
    
#     Rmat = np.concatenate((Vx,Vy,Vz), axis=1) # Rotation matrix
    
#     print (Rmat)
#     return Rmat, Vx,Vy,Vz

# def ZRotationMatrix(theta):
#     sint = np.sin(theta) # sin theta
#     cost = np.cos(theta) # cos theta
#     return np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])


# def ComputeAngle(base_link_vector,following_link_vector):
    
#     '''
#     Parameters: base_link_vector,following_link_vector -> 3D vectors representing a human segment
    
#     Output: Angle in btw the two vectors
#     '''
#     cross_product = np.cross(base_link_vector,following_link_vector)
#     sin_theta = np.absolute(cross_product)/(np.absolute(base_link_vector)*np.absolute(following_link_vector))
#     theta = np.arcsin(sin_theta)
    
#     return theta

def myRollWrap(angle):
    if angle < 0 :
        roll = angle + 180
    else:
        roll = angle - 180
    
    return roll
            

def ImageCoordinateFrame(image):
    cv2.line(image, np.array([0,0]), np.array([20,0]), (255, 0, 0), 3)
    cv2.line(image, np.array([0,0]), np.array([0,20]), (0, 0, 255), 3)
    
    cv2.putText(image, 'x', (30,25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, red, 2)
    cv2.putText(image, 'y', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, green, 2)

def _3DCoordinateFrame(image, _2D_Origin, _3D_Origin, rot_vec, trans_vec, cam_matrix, dist_matrix):
    
    VbaseX, _ = cv2.projectPoints((_3D_Origin[0]+300, _3D_Origin[1], _3D_Origin[2]), rot_vec, trans_vec, cam_matrix, dist_matrix)
    VbaseY, _ = cv2.projectPoints((_3D_Origin[0], _3D_Origin[1]+300, _3D_Origin[2]), rot_vec, trans_vec, cam_matrix, dist_matrix)
    VbaseZ, _ = cv2.projectPoints((_3D_Origin[0], _3D_Origin[1], _3D_Origin[2]+300), rot_vec, trans_vec, cam_matrix, dist_matrix)
    
    cv2.line(image, np.array([_2D_Origin[0] , _2D_Origin[1]], dtype=int), np.array([VbaseX[0][0][0] , VbaseX[0][0][1]], dtype=int), red, 3)
    cv2.line(image, np.array([_2D_Origin[0] , _2D_Origin[1]], dtype=int), np.array([VbaseY[0][0][0] , VbaseY[0][0][1]], dtype=int), green, 3)
    cv2.line(image, np.array([_2D_Origin[0] , _2D_Origin[1]], dtype=int), np.array([VbaseZ[0][0][0] , VbaseZ[0][0][1]], dtype=int), blue, 3)
    
def normalize(vect):
    return vect/np.linalg.norm(vect)

def from_image_name_2_excel_row_value(file):
    removingPNGextensions = file[:-4]
    i = 1
    while (removingPNGextensions[-(i+1)] != '/' and removingPNGextensions[-(i+1)] != '\\'):
      i += 1
    try:
      string_int = int(removingPNGextensions[-i:])
      return string_int
    except ValueError:
      # Handle the exception
      print(bcolors.FAIL + 'Failed to convert iamge name into excel row value')
      return -1

def faceModel3D():
    
    face_3d = []
    
    # Get the 3D Coordinates   
    face_3d.append([0, 0, 0]);          # Nose tip
    face_3d.append([225, 170, -135]);   # Left eye left corner
    face_3d.append([-225, 170, -135]);  # Right eye right corner
    face_3d.append([340, 0, -270]);     # Left ear
    face_3d.append([-340, 0, -270]);    # Right ear
    face_3d.append([150, -150, -125]);  # Left Mouth corner
    face_3d.append([-150, -150, -125]);  # Right Mouth corner

    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64)
    
    return face_3d
            