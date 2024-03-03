import cv2
import mediapipe as mp
import numpy as np
import time
from sm_02_GUI import Gui
from sm_05_Pose2Angles import Pose2Angles
from sm_06_PoseEstimator import PoseEstimator
from sm_00_utils import ComputeAngle, ImageCoordinateFrame, _3DCoordinateFrame, myRollWrap, computeMidPosition, fromWorldLandmark2nparray, PoseLandmark
from sm_03_camera_calibration import camera_calibration
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sm_04_logger import Result

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            
folder_path = 'C:/Users/aless/OneDrive/Desktop/SafeMove/'
video = 'video_05'

cap = cv2.VideoCapture(folder_path + "videos/" + video + ".mp4") # 0 for webcam
# cap = cv2.VideoCapture(0) # 0 for webcam


save_pictures_in_excel = False
frames_to_skip = 1
fps_input_video = 30
period_btw_frames = 1/fps_input_video
# print(period_btw_frames)
# print(range(frames_to_skip))


res = Result(folder_path=folder_path, source_video=video)

count = 0 

NN = mp.solutions.pose

NN.Pose(static_image_mode=False,
                        model_complexity=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=1.0)

with NN.Pose() as PoseNN: # very important for the sake of computation efficiency, not sure why though.
    try: 
        PoseEstimator_ = PoseEstimator(PoseNN)
        PoseEstimator_.setBaricenterLimit(0.8) # value btw 0.0 and 1.0, where 0.0 means you are on one foot any time the hip projection is not in the middle of your feet, and 1.0 you are considered on one foot only if your hip ground projection is over your feet position.
        PoseEstimator_.setMaxKneeDifference(20) # maximum difference in your knee angles before you are considered on one foot

        while cap.isOpened():
            for i in range(frames_to_skip):
                success, image = cap.read() 
                if success:
                    count = count + 1    

            start = time.time()

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            angle_list = PoseEstimator_.run(image)
            
            
            time_stamp = count*period_btw_frames
            
            
            res.error_list.loc[len(res.error_list.index)] = time_stamp, angle_list
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
            
            PoseEstimator_.Gui_.draw3D(results.pose_world_landmarks)
            
            if ((count%50) == 0 or count == 0) and save_pictures_in_excel:
                res.add_picture(image,time_stamp, count)
            
            plt.pause(.001)

            if cv2.waitKey(5) & 0xFF == 27:
                break


        cap.release()
        
    finally:
        res.save_excel(res.error_list)
