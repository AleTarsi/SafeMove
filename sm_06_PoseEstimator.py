import cv2
import mediapipe as mp
import numpy as np
import time
from sm_02_GUI import Gui
from sm_05_Pose2Angles import Pose2Angles
from sm_00_utils import faceModel3D, ImageCoordinateFrame, Face3DCoordinateFrame, myRollWrap, computeMidPosition, fromWorldLandmark2nparray, PoseLandmark, HandLandmark
from sm_03_camera_calibration import camera_calibration
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sm_04_ResultsLogger import ResultsLogger

class PoseEstimator:
    
    def __init__(self, height, width):
        self.img_h = height
        self.img_w = width
        
    def setBaricenterLimit(self, value):
        """
        Just one definition is sufficient here
        """
        self.min_baricenter_position = 0.5 - value/2
        self.max_baricenter_position = value/2 + 0.5
        
    def setMaxKneeDifference(self, value):
        self.max_knee_difference = value
        
    
    def run(self, image, results, visualizePose=True):
        
        ############################ Extraction Phase  ####################################      
        if results.pose_landmarks:
            
            face_2d = []
            
            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark      
            
            idx = PoseLandmark.NOSE
            nose_2d = (landmarks[idx].x*self.img_w, landmarks[idx].y*self.img_h)
            nose = fromWorldLandmark2nparray(world_landmarks[idx])
            face_2d.append([landmarks[idx].x*self.img_w, landmarks[idx].y*self.img_h])
            
            idx = PoseLandmark.LEFT_EYE_OUTER
            face_2d.append([landmarks[idx].x*self.img_w, landmarks[idx].y*self.img_h])
            leftEyeOuter = fromWorldLandmark2nparray(world_landmarks[idx])
            
            idx = PoseLandmark.RIGHT_EYE_OUTER 
            face_2d.append([landmarks[idx].x*self.img_w, landmarks[idx].y*self.img_h])
            rightEyeOuter = fromWorldLandmark2nparray(world_landmarks[idx])
                
            idx = PoseLandmark.LEFT_EAR
            face_2d.append([landmarks[idx].x*self.img_w, landmarks[idx].y*self.img_h])
            leftEar = fromWorldLandmark2nparray(world_landmarks[idx])
            
            idx = PoseLandmark.RIGHT_EAR
            face_2d.append([landmarks[idx].x*self.img_w, landmarks[idx].y*self.img_h])
            rightEar = fromWorldLandmark2nparray(world_landmarks[idx])
            
            idx = PoseLandmark.MOUTH_LEFT
            face_2d.append([landmarks[idx].x*self.img_w, landmarks[idx].y*self.img_h])
            leftMouth = fromWorldLandmark2nparray(world_landmarks[idx])
            
            idx = PoseLandmark.MOUTH_RIGHT
            face_2d.append([landmarks[idx].x*self.img_w, landmarks[idx].y*self.img_h])
            rightMouth = fromWorldLandmark2nparray(world_landmarks[idx])
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
                    
            leftHip = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_HIP])
            rightHip = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_HIP])
            
            try:
                Hip = computeMidPosition(leftHip,rightHip)[0]
            except:
                print("Some problems computing Hip pose")
                    
            leftShoulder = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_SHOULDER])
            rightShoulder = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_SHOULDER])
            
            try:
                '''
                Observations: It seems that the point describing the chest tends to be leaned even when the person is standing straight 
                '''
                Chest = computeMidPosition(leftShoulder,rightShoulder)[0]
            except:
                print("Some problems computing Shoulder pose")
            
            rightElbow = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_ELBOW])
            leftElbow = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_ELBOW])
            rightWrist = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_WRIST])
            leftWrist = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_WRIST])
                
            leftKnee = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_KNEE])
            rightKnee = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_KNEE])
            leftAnkle = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_ANKLE])
            rightAnkle = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_ANKLE])
                
            ###########################   Angle computation Phase and subPlots ###########################################      

            waist_xaxis, waist_yaxis, waist_zaxis = Pose2Angles.BodyAxes(leftHip = leftHip)
            # self.Gui_.BodyReferenceFrame(waist_xaxis, waist_yaxis, waist_zaxis)
            
            chest_xaxis, chest_yaxis, chest_zaxis = Pose2Angles.BackAxes(left_shoulder_point=leftShoulder, chest=Chest)
            # self.Gui_.ChestReferenceFrame(chest_xaxis, chest_yaxis, chest_zaxis, chest=Chest)
            # self.Gui_.DrawTrunk(trunk_point=[Chest,Hip,leftHip,rightHip])
            
            chest_LR, chest_FB, chest_Rot = Pose2Angles.BackAngles(waist_xaxis, waist_yaxis, waist_zaxis, chest_xaxis, chest_yaxis, chest_zaxis)
            
            rs_flexion_FB, rs_abduction_CWCCW, ls_flexion_FB, ls_abduction_CCWCW = Pose2Angles.ShoulderAngles(rightShoulder,rightElbow,leftShoulder,leftElbow,chest_zaxis, chest_xaxis)
            # self.Gui_.DrawElbowLine(rightShoulder,rightElbow,leftShoulder,leftElbow)
            
            re_flexion, le_flexion = Pose2Angles.ElbowAngles(rightShoulder,rightElbow, rightWrist, leftShoulder, leftElbow, leftWrist)
            # self.Gui_.DrawWristLine(rightWrist,rightElbow,leftWrist,leftElbow)
            
            if results.right_hand_landmarks:                              
                r_hand_landmarks = results.right_hand_landmarks.landmark       
                
                rightPinkyKnuckle = fromWorldLandmark2nparray(r_hand_landmarks[HandLandmark.PINKY_MCP])
                rightIndexKnucle = fromWorldLandmark2nparray(r_hand_landmarks[HandLandmark.INDEX_FINGER_MCP])
                
                try:
                    rightHand = computeMidPosition(rightPinkyKnuckle,rightIndexKnucle)[0]
                except:
                    print("Some problems computing R. Hand pose")
                
                rw_flexion_UD, re_rotation_PS, rw_rotation_UR, rightWristLine, rightPalmLine, rightOrthogonalPalmLine = Pose2Angles.WristAngles(rightElbow, rightWrist, rightHand, rightIndexKnucle, rightPinkyKnuckle)
                # self.Gui_.DrawHandLine(rightWrist,rightHand,leftWrist,leftHand)
                # self.Gui_.DrawHandaxes(rightWrist,rightWristLine,rightPalmLine,rightOrthogonalPalmLine)        
                
            else:
                rw_flexion_UD, re_rotation_PS, rw_rotation_UR = np.zeros(3)
                
            
            if results.left_hand_landmarks:
                l_hand_landmarks = results.left_hand_landmarks.landmark
                
                leftPinkyKnuckle = fromWorldLandmark2nparray(l_hand_landmarks[HandLandmark.PINKY_MCP])
                leftIndexKnucle = fromWorldLandmark2nparray(l_hand_landmarks[HandLandmark.INDEX_FINGER_MCP])
                
                try:
                    leftHand = computeMidPosition(leftPinkyKnuckle,leftIndexKnucle)[0]
                except:
                    print("Some problems computing L. Hand pose")
            
                lw_flexion_UD, le_rotation_PS, lw_rotation_UR, leftWristLine, leftPalmLine, leftOrthogonalPalmLine = Pose2Angles.WristAngles(leftElbow, leftWrist, leftHand, leftIndexKnucle, leftPinkyKnuckle)
                # self.Gui_.DrawHandLine(rightWrist,rightHand,leftWrist,leftHand)
                # self.Gui_.DrawHandaxes(leftWrist,leftWristLine,leftPalmLine,leftOrthogonalPalmLine)     
                
            else:
                lw_flexion_UD, le_rotation_PS, lw_rotation_UR = np.zeros(3)
            
            rk_flexion, lk_flexion = Pose2Angles.KneeAngles(rightKnee, leftKnee, rightHip, leftHip, rightAnkle, leftAnkle)
            # self.Gui_.DrawKneeLine(rightKnee, leftKnee, rightHip, leftHip)
            # self.Gui_.DrawFootLine(rightKnee, leftKnee, rightAnkle, leftAnkle)
            contact_points, knee_difference = Pose2Angles.ComputeContactPoints(rk_flexion, lk_flexion, self.max_knee_difference, rightAnkle, leftAnkle, Hip, self.min_baricenter_position, self.max_baricenter_position)
            # self.Gui_.DrawBaricenterLine(rightAnkle, leftAnkle, Hip)

            # Estimation of the camera parameters
            focal_length, cam_matrix, dist_matrix = camera_calibration(self.img_h,self.img_w)

            face_3d = faceModel3D()
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_SQPNP) # SOLVEPNP_ITERATIVE Iterative method is based on Levenberg-Marquardt optimization. In this case, the function finds such a pose that minimizes reprojection error, that is the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints .

            # Display the nose direction - Project in the image plane a point far in front of the nose
            # face_3d[0][0], face_3d[0][1], face_3d[0][2]+3000 -> means a point placed further in the z direction in the nose reference frame
            nose_3d_projection, jacobian = cv2.projectPoints((face_3d[0][0], face_3d[0][1], face_3d[0][2]+1000), rot_vec, trans_vec, cam_matrix, dist_matrix)
            
            face_3d = np.concatenate((face_3d, np.array(([[face_3d[0][0], face_3d[0][1], face_3d[0][2]+100]]))), axis=0)

            # self.Gui_.Draw3DFace(face_3d) # one 
            
            Rmat,_ = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(Rmat) # this function implements the results written in the paper RotationMatrixToRollPitchYaw
            
            ###################################### Modify the image ##############################
            
            p1 = np.array([nose_2d[0], nose_2d[1]], dtype=int)
            p2 = np.array([nose_3d_projection[0][0][0] , nose_3d_projection[0][0][1]], dtype=int)
                    
            for idx, point in enumerate(face_2d):
                cv2.circle(image, (int(point[0]), int(point[1])), 3, (0,0,255), 3)
                    
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            
            ImageCoordinateFrame(image)
            
            _3Dorigin = np.array([face_3d[0][0], face_3d[0][1], face_3d[0][2]])
            _2Dorigin = np.array([nose_2d[0], nose_2d[1]], dtype=int)
            
            Face3DCoordinateFrame(image, _2Dorigin ,_3Dorigin, rot_vec, trans_vec, cam_matrix, dist_matrix)
                    
            
            # cv2.putText(image, f'rw_rotation_UR: {np.round(rw_rotation_UR,1)}', (20,420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            # cv2.putText(image, f'contact points: {contact_points}', (20,380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            # cv2.putText(image, f'chest_Rot: {np.round(chest_Rot,1)}', (20,340), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                     
            
            return angles[1], myRollWrap(angles[0]), angles[2],chest_LR,chest_FB,chest_Rot,rs_flexion_FB, rs_abduction_CWCCW, ls_flexion_FB, ls_abduction_CCWCW,re_flexion,re_rotation_PS,le_flexion,le_rotation_PS,rw_flexion_UD, rw_rotation_UR, lw_flexion_UD, lw_rotation_UR, rk_flexion, lk_flexion,contact_points,0,0 # index gives us the # of row present in the dataframe, we are writing in a new row the new value of the fields
        
        else:
            
            return -1
