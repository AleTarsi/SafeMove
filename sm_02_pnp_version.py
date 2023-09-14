import cv2
import mediapipe as mp
import numpy as np
from sm_01_utils import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

mp_pose.Pose(static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



###################################### Select Source ##########################################
cap = cv2.VideoCapture(0)

################################## Start Processing ###########################################
with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB as cv2 and mp use different standards
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape # height, width, channels
        
        ############################ Camera matrix and distortion martrix ################################
        focal_length = w * 1
        center = (w/2, h/2)
        camera_matrix = np.array(
                                    [[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype = "double"
                                    )
        
        distortion = np.zeros((4, 1))
        ####################################################################################################
        
        image.flags.writeable = False #make it computationally efficient
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            lm = results.pose_landmarks.landmark
            pl = mp_pose.PoseLandmark
            
            # Get coordinates

            elbow = fromLandMarkTo3dPose(pl.LEFT_ELBOW.value, lm, width=w, height=h)
            wrist = fromLandMarkTo3dPose(pl.LEFT_WRIST.value, lm, width=w, height=h)
            leftHip = fromLandMarkTo3dPose(pl.LEFT_HIP.value, lm, width=w, height=h)
            rightHip = fromLandMarkTo3dPose(pl.RIGHT_HIP.value, lm, width=w, height=h)
            leftShoulder = fromLandMarkTo3dPose(pl.LEFT_SHOULDER.value, lm, width=w, height=h)
            rightShoulder = fromLandMarkTo3dPose(pl.RIGHT_SHOULDER.value, lm, width=w, height=h)
            
            Hip = computeMidPosition(leftHip,rightHip)
            Neck = computeMidPosition(leftShoulder,rightShoulder)
            Chest = computeMidPosition(Hip,Neck)
            

            # Visualize angle
            cv2.circle(image, (int(Hip[0]),int(Hip[1])), radius=5, color=(0, 0, 255 ), thickness=3)
            cv2.circle(image, (int(Neck[0]),int(Neck[1])), radius=5, color=(0, 0, 255), thickness=3)
            # cv2.circle(image, (int(Chest[0]),int(Chest[1])), radius=5, color=(255, 255, 255), thickness=3)
            # cv2.line(image, Neck[0:1], Hip[0:1], (255, 0, 0), 3)
            # cv2.line(image, p1, p2, (255, 0, 0), 3)

            model_points = []
            image_points = []
            
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if idx<=10: ## Face and shoulder landmarks
                    model_points.append([lm.x*w, lm.y*h, lm.z])
                    image_points.append([lm.x * w, lm.y * h])
                    # print("idx ", idx, "lm.x", "lm.y", lm.y, "lm.z", lm.z, "\n")
            
            model_points=np.float64(model_points)
            image_points=np.float64(image_points)
            
            success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, distortion)
            
            rmat, jac = cv2.Rodrigues(rot_vec)
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            # print("x ", x, "y", y, "z", z, "\n")
            
            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"
                
            text = str(x)
            #  Add the text on the image
            cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text = str(y)
            #  Add the text on the image
            cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            text = str(z)
            #  Add the text on the image
            cv2.putText(image, text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            transformation = np.eye(4)  # needs to 4x4 because you have to use homogeneous coordinates
            transformation[0:3, 3] = trans_vec.squeeze()
            transformation[0:3, 0:3] = rmat
            
            
            # transform model coordinates into homogeneous coordinates
            model_points_hom = np.concatenate((model_points, np.ones((model_points.shape[0], 1))), axis=1)

            # apply the transformation
            world_points = model_points_hom.dot(np.linalg.inv(transformation).T)
                       

            
        except:
            print("Error :( \n")
            
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
