import cv2
import mediapipe as mp
import numpy as np
import time
from sm_02_GUI import Gui
from sm_05_Pose2Angles import Pose2Angles
from sm_06_PoseEstimator import PoseEstimator
from sm_00_utils import ImageCoordinateFrame, _3DCoordinateFrame, myRollWrap, computeMidPosition, fromWorldLandmark2nparray, computeFPS, PoseLandmark
from sm_03_camera_calibration import camera_calibration
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sm_04_ResultsLogger import ResultsLogger
            
            
save_pictures_in_excel = False
speed_up_rate = 1 # integer value greater than 1, it set the number of frames to skip in the video btw one computation and another
fps_input_video = 30
period_btw_frames = 1/fps_input_video
count = 0 


######################################## HYPER PARAMETERS ######################################################
max_baricenter_position = 0.9
min_baricenter_position = 0.1
max_knee_difference = 20
###############################################################################################################

# print(period_btw_frames)
# print(range(frames_to_skip))




mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

firstFigure = Gui() # Change figure parameter to increase the sieze of the video's window
angleDetective = Pose2Angles()
            
folder_path = 'C:/Users/aless/OneDrive/Desktop/SafeMove/'
video = 'video_05'

cap = cv2.VideoCapture(folder_path + "videos/" + video + ".mp4") # 0 for webcam
# cap = cv2.VideoCapture(0) # 0 for webcam


logger = ResultsLogger(folder_path=folder_path, source_video=video)
NN = mp.solutions.pose

NN.Pose(static_image_mode=False,
                        model_complexity=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=1.0)

with NN.Pose() as PoseNN: # very important for the sake of computation efficiency, not sure why though.
    try: 
        PoseEstimator_ = PoseEstimator()
        PoseEstimator_.setBaricenterLimit(0.8) # value btw 0.0 and 1.0, where 0.0 means you are on one foot any time the hip projection is not in the middle of your feet, and 1.0 you are considered on one foot only if your hip ground projection is over your feet position.
        PoseEstimator_.setMaxKneeDifference(20) # maximum difference in your knee angles before you are considered on one foot

        while cap.isOpened():
            for i in range(speed_up_rate):
                success, image = cap.read() 
                if success:
                    count = count + 1    

            start = time.time()

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False
            
            # Get the result
            results = PoseNN.process(image)
            
            # To improve performance
            image.flags.writeable = True
            
            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            
            face_3d = []
            face_2d = []

            ############################ Extraction Phase  ####################################      
            if results.pose_landmarks:
                
                landmarks = results.pose_landmarks.landmark
                world_landmarks = results.pose_world_landmarks.landmark                    
                
                idx = PoseLandmark.NOSE
                nose_2d = (landmarks[idx].x*img_w, landmarks[idx].y*img_h)
                nose = fromWorldLandmark2nparray(world_landmarks[idx])
                face_2d.append([landmarks[idx].x*img_w, landmarks[idx].y*img_h])
                
                idx = PoseLandmark.LEFT_EYE_OUTER
                face_2d.append([landmarks[idx].x*img_w, landmarks[idx].y*img_h])
                leftEyeOuter = fromWorldLandmark2nparray(world_landmarks[idx])
                
                idx = PoseLandmark.RIGHT_EYE_OUTER 
                face_2d.append([landmarks[idx].x*img_w, landmarks[idx].y*img_h])
                rightEyeOuter = fromWorldLandmark2nparray(world_landmarks[idx])
                    
                idx = PoseLandmark.LEFT_EAR
                face_2d.append([landmarks[idx].x*img_w, landmarks[idx].y*img_h])
                leftEar = fromWorldLandmark2nparray(world_landmarks[idx])
                
                idx = PoseLandmark.RIGHT_EAR
                face_2d.append([landmarks[idx].x*img_w, landmarks[idx].y*img_h])
                rightEar = fromWorldLandmark2nparray(world_landmarks[idx])
                
                idx = PoseLandmark.MOUTH_LEFT
                face_2d.append([landmarks[idx].x*img_w, landmarks[idx].y*img_h])
                leftMouth = fromWorldLandmark2nparray(world_landmarks[idx])
                
                idx = PoseLandmark.MOUTH_RIGHT
                face_2d.append([landmarks[idx].x*img_w, landmarks[idx].y*img_h])
                rightMouth = fromWorldLandmark2nparray(world_landmarks[idx])
                        
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
                rightPinky = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_PINKY])
                rightIndex = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_INDEX])
                
                try:
                    rightHand = computeMidPosition(rightPinky,rightIndex)[0]
                except:
                    print("Some problems computing R. Hand pose")
                
                leftPinky = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_PINKY])
                leftIndex = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_INDEX])
                
                try:
                    leftHand = computeMidPosition(leftPinky,leftIndex)[0]
                except:
                    print("Some problems computing L. Hand pose")
                    
                leftKnee = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_KNEE])
                rightKnee = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_KNEE])
                leftAnkle = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.LEFT_ANKLE])
                rightAnkle = fromWorldLandmark2nparray(world_landmarks[PoseLandmark.RIGHT_ANKLE])
                                                    
                ###########################   Angle computation Phase and subPlots ###########################################      
                firstFigure.ax.cla()
                firstFigure.ax.set_xlim3d(-1, 1)
                firstFigure.ax.set_ylim3d(-1, 1)
                firstFigure.ax.set_zlim3d(-1, 1)
                
                waist_xaxis, waist_yaxis, waist_zaxis = angleDetective.BodyAxes(leftHip = leftHip)
                # firstFigure.BodyReferenceFrame(waist_xaxis, waist_yaxis, waist_zaxis)
                
                chest_xaxis, chest_yaxis, chest_zaxis = angleDetective.BackAxes(left_shoulder_point=leftShoulder, chest=Chest)
                # firstFigure.ChestReferenceFrame(chest_xaxis, chest_yaxis, chest_zaxis, chest=Chest)
                # firstFigure.DrawTrunk(trunk_point=[Chest,Hip,leftHip,rightHip])
                
                chest_LR, chest_FB, chest_Rot = angleDetective.BackAngles(waist_xaxis, waist_yaxis, waist_zaxis, chest_xaxis, chest_yaxis, chest_zaxis)
                
                rs_flexion_FB, rs_abduction_CWCCW, ls_flexion_FB, ls_abduction_CCWCW = angleDetective.ShoulderAngles(rightShoulder,rightElbow,leftShoulder,leftElbow,chest_zaxis, chest_xaxis)
                # firstFigure.DrawElbowLine(rightShoulder,rightElbow,leftShoulder,leftElbow)
                
                re_flexion, le_flexion = angleDetective.ElbowAngles(rightShoulder,rightElbow, rightWrist, leftShoulder, leftElbow, leftWrist)
                # firstFigure.DrawWristLine(rightWrist,rightElbow,leftWrist,leftElbow)
                
                rw_flexion_UD, lw_flexion_UD, leftWristLine, leftPalmLine, leftOrthogonalPalmLine, rightWristLine, rightPalmLine, rightOrthogonalPalmLine = angleDetective.WristAngles(rightElbow, rightWrist, rightHand, rightIndex, rightPinky, leftElbow, leftWrist, leftHand, leftIndex, leftPinky)
                # firstFigure.DrawHandLine(rightWrist,rightHand,leftWrist,leftHand)
                # firstFigure.DrawHandaxes(leftWrist,leftWristLine,leftPalmLine,leftOrthogonalPalmLine)
                # firstFigure.DrawHandaxes(rightWrist,rightWristLine,rightPalmLine,rightOrthogonalPalmLine)        
                
                rk_flexion, lk_flexion = angleDetective.KneeAngles(rightKnee, leftKnee, rightHip, leftHip, rightAnkle, leftAnkle)
                # firstFigure.DrawKneeLine(rightKnee, leftKnee, rightHip, leftHip)
                # firstFigure.DrawFootLine(rightKnee, leftKnee, rightAnkle, leftAnkle)
                contact_points, knee_difference = angleDetective.ComputeContactPoints(rk_flexion, lk_flexion, max_knee_difference, rightAnkle, leftAnkle, Hip, min_baricenter_position, max_baricenter_position)
                # firstFigure.DrawBaricenterLine(rightAnkle, leftAnkle, Hip)
                
                # Get the 3D Coordinates   
                face_3d.append([0, 0, 0]);          # Nose tip
                face_3d.append([225, 170, -135]);   # Left eye left corner
                face_3d.append([-225, 170, -135]);  # Right eye right corner
                face_3d.append([340, 0, -270]);     # Left ear
                face_3d.append([-340, 0, -270]);    # Right ear
                face_3d.append([150, -150, -125]);  # Left Mouth corner
                face_3d.append([-150, -150, -125]);  # Right Mouth corner
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)


                # Estimation of the camera parameters
                focal_length, cam_matrix, dist_matrix = camera_calibration(img_h,img_w)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_SQPNP) # SOLVEPNP_ITERATIVE Iterative method is based on Levenberg-Marquardt optimization. In this case, the function finds such a pose that minimizes reprojection error, that is the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints .

                # Display the nose direction - Project in the image plane a point far in front of the nose
                # face_3d[0][0], face_3d[0][1], face_3d[0][2]+3000 -> means a point placed further in the z direction in the nose reference frame
                nose_3d_projection, jacobian = cv2.projectPoints((face_3d[0][0], face_3d[0][1], face_3d[0][2]+1000), rot_vec, trans_vec, cam_matrix, dist_matrix)
                
                # _3DCoordinateFrame(image, rot_vec, trans_vec, cam_matrix, dist_matrix) 

                face_3d = np.concatenate((face_3d, np.array(([[face_3d[0][0], face_3d[0][1], face_3d[0][2]+100]]))), axis=0)

                Rmat,_ = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(Rmat) # this function implements the results written in the paper RotationMatrixToRollPitchYaw
                
                # firstFigure.Draw3DFace(face_3d) # one 
                
                p1 = np.array([nose_2d[0], nose_2d[1]], dtype=int)
                p2 = np.array([nose_3d_projection[0][0][0] , nose_3d_projection[0][0][1]], dtype=int)
                        
                for idx, point in enumerate(face_2d):
                    cv2.circle(image, (int(point[0]), int(point[1])), 3, (0,0,255), 3)
                        
                cv2.line(image, p1, p2, (255, 0, 0), 3)
                
                ImageCoordinateFrame(image)
                
                _3Dorigin = np.array([face_3d[0][0], face_3d[0][1], face_3d[0][2]])
                _2Dorigin = np.array([nose_2d[0], nose_2d[1]], dtype=int)
                
                _3DCoordinateFrame(image, _2Dorigin ,_3Dorigin, rot_vec, trans_vec, cam_matrix, dist_matrix)
                
                cv2.putText(image, f't: {np.round(knee_difference,1)}', (20,420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                cv2.putText(image, f'contact points: {contact_points}', (20,380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                # cv2.putText(image, f'chest_Rot: {np.round(chest_Rot,1)}', (20,340), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

                time_stamp = count*period_btw_frames
                logger.error_list.loc[len(logger.error_list.index)] = [time_stamp, angles[1], myRollWrap(angles[0]), angles[2],chest_LR,chest_FB,chest_Rot,rs_flexion_FB, rs_abduction_CWCCW, ls_flexion_FB, ls_abduction_CCWCW,re_flexion,0,le_flexion,0,rw_flexion_UD, 0,lw_flexion_UD,0, rk_flexion, lk_flexion,contact_points,0,0] # index gives us the # of row present in the dataframe, we are writing in a new row the new value of the fields
                # print(res.error_list)
                
                
                end = time.time()
                fps = computeFPS(end,start,speed_up_rate)

                cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, NN.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )

                cv2.imshow('Head Pose Estimation', image)
                
                firstFigure.draw3D(results.pose_world_landmarks)
                
                if save_pictures_in_excel:
                    logger.add_picture(image,time_stamp, count, PicturesamplingTime=50)
                
                plt.pause(.001)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        
    finally:
        logger.save_excel(logger.error_list)
