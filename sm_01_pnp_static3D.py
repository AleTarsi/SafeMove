import cv2
import mediapipe as mp
import numpy as np
import time
from sm_02_GUI import Gui
from sm_05_computation import Computation
from sm_00_utils import ComputeAngle, ImageCoordinateFrame, _3DCoordinateFrame, myRollWrap, computeMidPosition, fromWorldLandmark2nparray, PoseLandmark
from sm_03_camera_calibration import camera_calibration
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sm_04_logger import Result

mp_pose = mp.solutions.pose

mp_pose.Pose(static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)


mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

firstFigure = Gui() # Change figure parameter to increase the sieze of the video's window
angleDetective = Computation()
            
path='C:/Users/aless/OneDrive/Desktop/SafeMove/videos/video_05.mp4'
cap = cv2.VideoCapture(path) # 0 for webcam


frames_to_skip = 1
fps_input_video = 30
period_btw_frames = 1/fps_input_video

res = Result(source_video=path)

count = 0 

# print(period_btw_frames)
# print(range(frames_to_skip))

with mp_pose.Pose() as pose: # very important for the sake of computation efficiency, not sure why though.
    while cap.isOpened():
        for i in range(frames_to_skip):
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
        results = pose.process(image)
        
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
                    
            idx = PoseLandmark.LEFT_HIP
            leftHip = fromWorldLandmark2nparray(world_landmarks[idx])
            
            idx = PoseLandmark.RIGHT_HIP
            rightHip = fromWorldLandmark2nparray(world_landmarks[idx])
            try:
                Hip = computeMidPosition(leftHip,rightHip)[0]
            except:
                print("Some problems computing Hip pose")
                    
            idx = PoseLandmark.LEFT_SHOULDER
            leftShoulder = fromWorldLandmark2nparray(world_landmarks[idx])
            idx = PoseLandmark.RIGHT_SHOULDER
            rightShoulder = fromWorldLandmark2nparray(world_landmarks[idx])
            try:
                '''
                Observations: It seems that the point describing the chest tends to be leaned even when the person is standing straight 
                '''
                Chest = computeMidPosition(leftShoulder,rightShoulder)[0]
            except:
                print("Some problems computing Shoulder pose")
            
            idx = PoseLandmark.RIGHT_ELBOW
            rightElbow = fromWorldLandmark2nparray(world_landmarks[idx])
            
            idx = PoseLandmark.LEFT_ELBOW
            leftElbow = fromWorldLandmark2nparray(world_landmarks[idx])
            
            idx = PoseLandmark.RIGHT_WRIST
            rightWrist = fromWorldLandmark2nparray(world_landmarks[idx])
            
            idx = PoseLandmark.LEFT_WRIST
            leftWrist = fromWorldLandmark2nparray(world_landmarks[idx])
                                                
            ##############################################################################      
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
                    
            Rmat,_ = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(Rmat) # this function implements the results written in the paper RotationMatrixToRollPitchYaw
            
                    
            # print('x: ',  myRollWrap(angles[0])) 
            # print('y: ',  angles[1]) 
            # print('z: ',  angles[2]) 
            
            cv2.putText(image, f'flexion_FB: {np.round(ls_flexion_FB,1)}', (20,420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            cv2.putText(image, f'CCWCW: {np.round(ls_abduction_CCWCW,1)}', (20,380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            # cv2.putText(image, f'chest_Rot: {np.round(chest_Rot,1)}', (20,340), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            
            res.error_list.loc[len(res.error_list.index)] = [count*period_btw_frames, angles[1], myRollWrap(angles[0]), angles[2],chest_LR,chest_FB,chest_Rot,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] # index gives us the # of row present in the dataframe, we are writing in a new row the new value of the fields
            # print(res.error_list)
            
            end = time.time()
            totalTime = end - start

            try:
                fps_output = (1 / totalTime)*frames_to_skip
            except:
                fps_output= -1

            cv2.putText(image, f'FPS: {int(fps_output)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )

            cv2.imshow('Head Pose Estimation', image)
            
            # firstFigure.draw3D(results.pose_world_landmarks)
            
            
            plt.pause(.001)

        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
