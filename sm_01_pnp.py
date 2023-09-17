import cv2
import mediapipe as mp
import numpy as np
import time
from sm_02_GUI import Gui
from sm_03_camera_calibration import camera_calibration

mp_pose = mp.solutions.pose

mp_pose.Pose(static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)


mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

my_GUI = Gui()

path='C:/Users/aless/OneDrive/Desktop/SafeMove/videos/video_06.mp4'
cap = cv2.VideoCapture(0)

while cap.isOpened():
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

    if results.pose_world_landmarks.landmark:
        for idx, lm in enumerate(results.pose_landmarks.landmark): #CHECK IF I NEED WORLD IN HERE!
            if idx <=10:
                if idx == 0:
                    nose_2d = (lm.x, lm.y)
                    nose_3d = (lm.x, lm.y, lm.z)
                    
                x, y = int(lm.x), int(lm.y)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])       
        
        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)
                
        my_GUI.draw3D(results.pose_world_landmarks)


        # Estimation of the camera parameters
        focal_length, cam_matrix, dist_matrix = camera_calibration(img_h,img_w)

        # Solve PnP
             
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE) # SOLVEPNP_ITERATIVE Iterative method is based on Levenberg-Marquardt optimization. In this case, the function finds such a pose that minimizes reprojection error, that is the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints .

        # # Get rotational matrix
        # rmat, jac = cv2.Rodrigues(rot_vec)

        # # Get angles
        # angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # # Get the y rotation degree
        # x = angles[0] * 360
        # y = angles[1] * 360
        # z = angles[2] * 360
        

        # Display the nose direction - Project in the image plane a point far in front of the nose
        nose_3d_projection, jacobian = cv2.projectPoints((nose_3d[0], nose_3d[1], nose_3d[2]*1000), rot_vec, trans_vec, cam_matrix, dist_matrix)

        p1 = np.array([nose_2d[0], nose_2d[1]])
        p2 = np.array([nose_3d_projection[0][0][0] , nose_3d_projection[0][0][1]])
        
        # cv2.line(image, int(p1), int(p2), (255, 0, 0), 3)
        cv2.circle(image, (int(p1[0]), int(p1[1])), 3, (0,0,255), 3)
        # I have the nose on the top left corner in this way
        cv2.putText(image, f'P1.X: {p1[0]}', (20,370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.putText(image, f'P1.Y: {p2[1]}', (20,410), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        # # Add the text on the image
        # text = str(x)
        # #  Add the text on the image
        # cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # text = str(y)
        # #  Add the text on the image
        # cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # text = str(z)
        # #  Add the text on the image
        # cv2.putText(image, text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
        # if y < -10:
        #     text = "Looking Left"
        # elif y > 10:
        #     text = "Looking Right"
        # elif x < -10:
        #     text = "Looking Down"
        # elif x > 10:
        #     text = "Looking Up"
        # else:
        #     text = "Forward"
        
        # cv2.putText(image, text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    end = time.time()
    totalTime = end - start

    try:
        fps = 1 / totalTime
    except:
        fps= -1

    cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    # Render detections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
    #                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
    #                         )

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()
