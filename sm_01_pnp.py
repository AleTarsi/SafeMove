import cv2
import mediapipe as mp
import numpy as np
import time
from sm_02_GUI import Gui
from sm_03_camera_calibration import camera_calibration

mp_pose = mp.solutions.pose

mp_pose.Pose(static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)


mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

my_GUI = Gui()

path='C:/Users/aless/OneDrive/Desktop/SafeMove/videos/video_07.mp4'
cap = cv2.VideoCapture(0)
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
        for idx, lm in enumerate(results.pose_landmarks.landmark): #CHECK IF I NEED WORLD IN HERE!
            world_lm = results.pose_world_landmarks.landmark[idx]
            if idx <=10:
                
                if idx == 0:
                    nose_2d = (lm.x*img_w, lm.y*img_h)
                    nose_3d = (-world_lm.x, -world_lm.y, -world_lm.z)
                
                # Get the 2D Coordinates
                face_2d.append([lm.x*img_w, lm.y*img_h])

                # Get the 3D Coordinates
                face_3d.append([-world_lm.x, -world_lm.y, -world_lm.z])       
        
        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)
                
        # my_GUI.draw3D(results.pose_world_landmarks)


        # Estimation of the camera parameters
        focal_length, cam_matrix, dist_matrix = camera_calibration(img_h,img_w)

        # Solve PnP
        
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_SQPNP) # SOLVEPNP_ITERATIVE Iterative method is based on Levenberg-Marquardt optimization. In this case, the function finds such a pose that minimizes reprojection error, that is the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints .

        transformation = np.eye(4)  # needs to 4x4 because you have to use homogeneous coordinates
        transformation[0:3, 3] = trans_vec.squeeze() # Homogeneous transformation from camera to world
        
        # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate

        # transform model coordinates into homogeneous coordinates
        model_points_hom = np.concatenate((face_3d, np.ones((11, 1))), axis=1)
        
        # apply the transformation
        world_points_CamFrame = model_points_hom.dot(np.linalg.inv(transformation).T)
        
        # my_GUI.drawRotationVector(rot_vec/ np.linalg.norm(rot_vec))
        
        # # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # # Get angles
        # angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # # Get the y rotation degree
        # x = angles[0] * 360
        # y = angles[1] * 360
        # z = angles[2] * 360
        

        # Display the nose direction - Project in the image plane a point far in front of the nose
        nose_3d_projection, jacobian = cv2.projectPoints((nose_3d[0], nose_3d[1]+10, nose_3d[2]), rot_vec, trans_vec, cam_matrix, dist_matrix)
        
        my_GUI.draw3D(results.pose_world_landmarks, extraPoint=[nose_3d[0], nose_3d[1], nose_3d[2]+np.dot(rmat[2][0:3],np.array([10,0,0]))])
        # my_GUI.draw3D(results.pose_world_landmarks, extraPoint=nose_3d4_projection)
        
        # The transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate
        # If you look at the 3d plot you will realize that the body does not translate in space, but rotates. Hence we just need a translation.
        
        
        p1 = np.array([nose_2d[0], nose_2d[1]], dtype=int)
        p2 = np.array([nose_3d_projection[0][0][0] , nose_3d_projection[0][0][1]], dtype=int)
        
        # cv2.line(image, int(p1), int(p2), (255, 0, 0), 3)
        
        for idx, point in enumerate(face_2d):
            cv2.circle(image, (int(point[0]), int(point[1])), 3, (0,0,255), 3)
                
        cv2.line(image, p1, p2, (255, 0, 0), 3)
        
        # I have the nose on the top left corner in this way
        cv2.putText(image, f'P1.X: {p1[0]}', (20,370), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.putText(image, f'P1.Y: {p1[1]}', (20,410), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

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
