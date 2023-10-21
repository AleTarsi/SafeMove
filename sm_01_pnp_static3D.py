import cv2
import mediapipe as mp
import numpy as np
import time
from sm_02_GUI import Gui
from sm_00_utils import ComputeAngle, ImageCoordinateFrame, _3DCoordinateFrame
from sm_03_camera_calibration import camera_calibration
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

mp_pose = mp.solutions.pose

mp_pose.Pose(static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)


mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

firstFigure = Gui()
            
path='C:/Users/aless/OneDrive/Desktop/SafeMove/videos/video_07.mp4'
cap = cv2.VideoCapture(0) # 0 for webcam
speed=5

while cap.isOpened():
    for i in range(speed):
        success, image = cap.read()     

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = mp_pose.Pose().process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    
    face_3d = []
    face_2d = []

    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            world_lm = results.pose_world_landmarks.landmark[idx]
            if idx == 0 or idx == 3 or idx == 6 or idx == 7 or idx == 8 or idx == 9 or idx == 10:
                
                if idx == 0:
                    nose_2d = (lm.x*img_w, lm.y*img_h)
                    nose_3d = (-world_lm.x, -world_lm.y, -world_lm.z)
                
                # Get the 2D Coordinates
                face_2d.append([lm.x*img_w, lm.y*img_h])
                

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
        nose_3d_projection, jacobian = cv2.projectPoints((face_3d[0][0], face_3d[0][1], face_3d[0][2]+3000), rot_vec, trans_vec, cam_matrix, dist_matrix)
        
        # _3DCoordinateFrame(image, rot_vec, trans_vec, cam_matrix, dist_matrix) 

        face_3d = np.concatenate((face_3d, np.array(([[face_3d[0][0], face_3d[0][1], face_3d[0][2]+100]]))), axis=0)

        firstFigure.Draw3DFace(face_3d) # one 
        
        p1 = np.array([nose_2d[0], nose_2d[1]], dtype=int)
        p2 = np.array([nose_3d_projection[0][0][0] , nose_3d_projection[0][0][1]], dtype=int)
                
        for idx, point in enumerate(face_2d):
            cv2.circle(image, (int(point[0]), int(point[1])), 3, (0,0,255), 3)
                
        cv2.line(image, p1, p2, (255, 0, 0), 3)
        
        ImageCoordinateFrame(image)
        
        print(rot_vec)
                
        # FnoseX,FnoseY,FnoseZ = Compute3DBaseGivenV1(face_3d[0][0], face_3d[0][1], face_3d[0][2]+3000)
        # I have the nose on the top left corner in this way
        # cv2.putText(image, f'angle: {ComputeAngle(v1,v2)}', (20,370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        
        

    end = time.time()
    totalTime = end - start

    try:
        fps = (1 / totalTime)*speed
    except:
        fps= -1

    cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                            )

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()
