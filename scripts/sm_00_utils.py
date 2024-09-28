# MIT License
#
# Copyright (c) [YEAR] [Your Name or Your Organization]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
  
class HandLandmark():
  """The 21 hand landmarks."""

  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20
  
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

def Face3DCoordinateFrame(image, _2D_Origin, _3D_Origin, rot_vec, trans_vec, cam_matrix, dist_matrix):
    
    VbaseX, _ = cv2.projectPoints((_3D_Origin[0]+300, _3D_Origin[1], _3D_Origin[2]), rot_vec, trans_vec, cam_matrix, dist_matrix)
    VbaseY, _ = cv2.projectPoints((_3D_Origin[0], _3D_Origin[1]+300, _3D_Origin[2]), rot_vec, trans_vec, cam_matrix, dist_matrix)
    VbaseZ, _ = cv2.projectPoints((_3D_Origin[0], _3D_Origin[1], _3D_Origin[2]+300), rot_vec, trans_vec, cam_matrix, dist_matrix)
    
    cv2.line(image, np.array([_2D_Origin[0] , _2D_Origin[1]], dtype=int), np.array([VbaseX[0][0][0] , VbaseX[0][0][1]], dtype=int), red, 3)
    cv2.line(image, np.array([_2D_Origin[0] , _2D_Origin[1]], dtype=int), np.array([VbaseY[0][0][0] , VbaseY[0][0][1]], dtype=int), green, 3)
    cv2.line(image, np.array([_2D_Origin[0] , _2D_Origin[1]], dtype=int), np.array([VbaseZ[0][0][0] , VbaseZ[0][0][1]], dtype=int), blue, 3)
    
def normalize(vect):
    try:
        return vect/np.linalg.norm(vect)
    except:
        return np.zeros(3)

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
            