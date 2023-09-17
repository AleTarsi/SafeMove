import cv2
import mediapipe as mp
import numpy as np
from sm_00_utils import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

mp_pose.Pose(static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

###################################### Select Source ##########################################
path='C:/Users/aless/OneDrive/Desktop/SafeMove/videos/video_06.mp4'
cap = cv2.VideoCapture(path)

################################## Start Processing ###########################################
with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB as cv2 and mp use different standards
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape # height, width, channels
        
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

            hip2neck = np.array(Neck - Hip)
            hip2vertical_line = np.array(np.array([Hip[0],100,Hip[2]]) - Hip)
            theta = np.degrees(np.arccos(np.dot(hip2neck,hip2vertical_line)/(np.linalg.norm(hip2neck)*np.linalg.norm(hip2vertical_line))))
            
            cv2.putText(image, str(int(theta)), 
                           tuple((int(Chest[0]),int(Chest[1]))), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
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
