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

import cv2
import mediapipe as mp
import numpy as np
import time
from sm_02_GUI import Gui
from sm_05_Pose2Angles import Pose2Angles
from sm_00_utils import faceModel3D, ImageCoordinateFrame, Plot3DCoordinateFrame, myRollWrap, computeMidPosition, worldLandmark2numpy, PoseLandmark, HandLandmark, computeChestFrame
from sm_03_camera_calibration import camera_calibration
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sm_04_ResultsLogger import ResultsLogger

class PoseEstimator:
    
    def __init__(self):
        self.setBaricenterLimit()
        self.setMaxKneeDifference()
        
    def setBaricenterLimit(self, value = 0.8):
        """
        Value btw 0.0 and 1.0, where 0.0 means you are on one foot any time the hip projection is not in the middle of your feet, and 1.0 you are considered on one foot only if your hip projection is outside your feet position.        
        """
        self.min_baricenter_position = 0.5 - value/2
        self.max_baricenter_position = value/2 + 0.5
        
    def setMaxKneeDifference(self, value = 20):
        """
        Maximum difference between your knees angles before you are considered on one foot. [degrees]
        """
        self.max_knee_difference = value
        
    
    def run(self, image:np.ndarray, results, gui:Gui):
        
        if results.pose_landmarks:
            
            img_h, img_w, _ = image.shape

            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark    
              
            ############################ Extraction Phase  ####################################      

            face_2d = []
            for idx in [PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR, PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT]:
                face_2d.append([landmarks[idx].x*img_w, landmarks[idx].y*img_h])
            
            face_2d = np.array(face_2d, dtype=np.float64) # Convert it to the NumPy array
                    
            leftHip = worldLandmark2numpy(world_landmarks[PoseLandmark.LEFT_HIP])
            rightHip = worldLandmark2numpy(world_landmarks[PoseLandmark.RIGHT_HIP])
            
            Hip = computeMidPosition(leftHip,rightHip)[0]
                    
            leftShoulder = worldLandmark2numpy(world_landmarks[PoseLandmark.LEFT_SHOULDER])
            rightShoulder = worldLandmark2numpy(world_landmarks[PoseLandmark.RIGHT_SHOULDER])
            
            Chest = computeMidPosition(leftShoulder,rightShoulder)[0]

            rightElbow = worldLandmark2numpy(world_landmarks[PoseLandmark.RIGHT_ELBOW])
            leftElbow = worldLandmark2numpy(world_landmarks[PoseLandmark.LEFT_ELBOW])
            rightWrist = worldLandmark2numpy(world_landmarks[PoseLandmark.RIGHT_WRIST])
            leftWrist = worldLandmark2numpy(world_landmarks[PoseLandmark.LEFT_WRIST])
                
            leftKnee = worldLandmark2numpy(world_landmarks[PoseLandmark.LEFT_KNEE])
            rightKnee = worldLandmark2numpy(world_landmarks[PoseLandmark.RIGHT_KNEE])
            leftAnkle = worldLandmark2numpy(world_landmarks[PoseLandmark.LEFT_ANKLE])
            rightAnkle = worldLandmark2numpy(world_landmarks[PoseLandmark.RIGHT_ANKLE])
                
            ###########################   Angle computation Phase and subPlots ###########################################      

            waist_xaxis, waist_yaxis, waist_zaxis = Pose2Angles.BodyAxes(leftHip)
            # gui.BodyReferenceFrame(waist_xaxis, waist_yaxis, waist_zaxis)
            
            chest_xaxis, chest_yaxis, chest_zaxis = Pose2Angles.BackAxes(left_shoulder_point=leftShoulder, chest=Chest, hip=Hip)
            # gui.ChestReferenceFrame(chest_xaxis, chest_yaxis, chest_zaxis, chest=Chest)
            # gui.DrawTrunk(trunk_point=[Chest,Hip,leftHip,rightHip])
            
            chest_LR, chest_FB, chest_Rot = Pose2Angles.BackAngles(waist_xaxis, waist_yaxis, waist_zaxis, chest_xaxis, chest_yaxis, chest_zaxis)
            
            rs_flexion_FB, rs_abduction_CWCCW, ls_flexion_FB, ls_abduction_CCWCW = Pose2Angles.ShoulderAngles(rightShoulder,rightElbow,leftShoulder,leftElbow,chest_zaxis, chest_xaxis)
            # gui.DrawElbowLine(rightShoulder,rightElbow,leftShoulder,leftElbow)
            
            re_flexion, le_flexion = Pose2Angles.ElbowAngles(rightShoulder,rightElbow, rightWrist, leftShoulder, leftElbow, leftWrist)
            # gui.DrawWristLine(rightWrist,rightElbow,leftWrist,leftElbow)
            
            if results.world_landmarks.multi_hand_world_landmarks:                              
                r_hand_landmarks = results.multi_hand_world_landmarks.landmark       
                
                rightPinkyKnuckle = worldLandmark2numpy(r_hand_landmarks[HandLandmark.PINKY_MCP])
                rightIndexKnucle = worldLandmark2numpy(r_hand_landmarks[HandLandmark.INDEX_FINGER_MCP])
                
                rightHand = computeMidPosition(rightPinkyKnuckle,rightIndexKnucle)[0]
                
                rw_flexion_UD, re_rotation_PS, rw_rotation_UR, rightWristLine, rightPalmLine, rightOrthogonalPalmLine = Pose2Angles.WristAngles(rightElbow, rightWrist, rightHand, rightIndexKnucle, rightPinkyKnuckle, waist_xaxis)
                       
                
            else:
                rw_flexion_UD, re_rotation_PS, rw_rotation_UR = np.zeros(3)
                
            
            if results.left_hand_landmarks:
                l_hand_landmarks = results.left_hand_landmarks.landmark
                
                leftPinkyKnuckle = worldLandmark2numpy(l_hand_landmarks[HandLandmark.PINKY_MCP])
                leftIndexKnucle = worldLandmark2numpy(l_hand_landmarks[HandLandmark.INDEX_FINGER_MCP])
                
                leftHand = computeMidPosition(leftPinkyKnuckle,leftIndexKnucle)[0]

                lw_flexion_UD, le_rotation_SP, lw_rotation_UR, leftWristLine, leftPalmLine, leftOrthogonalPalmLine = Pose2Angles.WristAngles(leftElbow, leftWrist, leftHand, leftIndexKnucle, leftPinkyKnuckle, waist_xaxis, left_flag=True)
                le_rotation_PS = - le_rotation_SP 
                
                
            else:
                lw_flexion_UD, le_rotation_PS, lw_rotation_UR = np.zeros(3)
            
            try:
                gui.DrawHandsLine(rightWrist,rightHand,leftWrist,leftHand)
                # gui.DrawHandaxes(leftWrist,leftWristLine,leftPalmLine,leftOrthogonalPalmLine)   
                # gui.DrawHandaxes(rightWrist,rightWristLine,rightPalmLine,rightOrthogonalPalmLine)
            except:
                pass
            
            rk_flexion, lk_flexion = Pose2Angles.KneeAngles(rightKnee, leftKnee, rightHip, leftHip, rightAnkle, leftAnkle)
            # gui.DrawKneeLine(rightKnee, leftKnee, rightHip, leftHip)
            # gui.DrawFootLine(rightKnee, leftKnee, rightAnkle, leftAnkle)
            contact_points, knee_difference = Pose2Angles.ComputeContactPoints(rk_flexion, lk_flexion, self.max_knee_difference, rightAnkle, leftAnkle, Hip, self.min_baricenter_position, self.max_baricenter_position)
            # gui.DrawBaricenterLine(rightAnkle, leftAnkle, Hip)

            # Estimation of the camera parameters
            focal_length, cam_matrix, dist_matrix = camera_calibration(img_h,img_w)

            chestFrame = computeChestFrame(Chest, leftShoulder, Hip)
            
            rot_vec = cv2.Rodrigues(chestFrame)[0]
            _2Dorigin = np.array([landmarks[PoseLandmark.RIGHT_SHOULDER].x*img_w, landmarks[PoseLandmark.RIGHT_SHOULDER].y*img_h], dtype=int)
            trans_vec = worldLandmark2numpy(world_landmarks[PoseLandmark.RIGHT_SHOULDER])
            # Plot3DCoordinateFrame(image, _2Dorigin, [0,0,0], rot_vec, trans_vec, cam_matrix, dist_matrix)
            
            face_3d = faceModel3D()
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_SQPNP) # SOLVEPNP_ITERATIVE Iterative method is based on Levenberg-Marquardt optimization. In this case, the function finds such a pose that minimizes reprojection error, that is the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints .

            # Display the nose direction - Project in the image plane a point far in front of the nose
            # face_3d[0][0], face_3d[0][1], face_3d[0][2]+3000 -> means a point placed further in the z direction in the nose reference frame
            nose_3d_projection, jacobian = cv2.projectPoints((face_3d[0][0], face_3d[0][1], face_3d[0][2]+1000), rot_vec, trans_vec, cam_matrix, dist_matrix)
            
            face_3d = np.concatenate((face_3d, np.array(([[face_3d[0][0], face_3d[0][1], face_3d[0][2]+100]]))), axis=0)
            # gui.Draw3DFace(face_3d) # one 
            
            Rmat,_ = cv2.Rodrigues(rot_vec)
            head_rotation_LR, head_flexion_DU, head_flexion_CCWCW = Pose2Angles.HeadAngles(Rmat, chestFrame)

            ###################################### Modify the image ##############################
            nose_2d = (landmarks[PoseLandmark.NOSE].x*img_w, landmarks[PoseLandmark.NOSE].y*img_h)
            p1 = np.array([nose_2d[0], nose_2d[1]], dtype=int)
            p2 = np.array([nose_3d_projection[0][0][0] , nose_3d_projection[0][0][1]], dtype=int)
                    
            for idx, point in enumerate(face_2d):
                cv2.circle(image, (int(point[0]), int(point[1])), 3, (0,0,255), 3)
                    
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            
            ImageCoordinateFrame(image)
            
            _3Dorigin = np.array([face_3d[0][0], face_3d[0][1], face_3d[0][2]])
            _2Dorigin = np.array([nose_2d[0], nose_2d[1]], dtype=int)
            
            Plot3DCoordinateFrame(image, _2Dorigin ,_3Dorigin, rot_vec, trans_vec, cam_matrix, dist_matrix)
                    
            return head_rotation_LR, head_flexion_DU, head_flexion_CCWCW, chest_LR,chest_FB,chest_Rot,rs_flexion_FB, rs_abduction_CWCCW, ls_flexion_FB, ls_abduction_CCWCW,re_flexion,re_rotation_PS,le_flexion,le_rotation_PS,rw_flexion_UD, rw_rotation_UR, lw_flexion_UD, lw_rotation_UR, rk_flexion, lk_flexion,contact_points # index gives us the # of row present in the dataframe, we are writing in a new row the new value of the fields
        
        else:
            
            return -1
