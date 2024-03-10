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
# print(period_btw_frames)
# print(range(speed_up_rate))

folder_path = 'C:/Users/aless/OneDrive/Desktop/SafeMove/'
video = 'video_05'

cap = cv2.VideoCapture(folder_path + "videos/" + video + ".mp4") # 0 for webcam
# cap = cv2.VideoCapture(0) # 0 for webcam

result = ResultsLogger(folder_path=folder_path, source_video=video)
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
            
            PoseEstimator_.set_image(image)
            
            angle_list = PoseEstimator_.run(results, visualizePose=True)
            
            if angle_list != -1:
                    
                time_stamp = count*period_btw_frames
                
                result.error_list.loc[len(result.error_list.index)] = [time_stamp, *angle_list] # the Asterisk unpack tuples or lists
                # print(result.error_list)
                
            end = time.time()
            fps = computeFPS(end,start,speed_up_rate)
            
            PoseEstimator_.Gui_.showText(PoseEstimator_.get_image(), f'FPS: {int(fps)}', (20,450))   
                
            cv2.imshow('Head Pose Estimation', PoseEstimator_.get_image())
            
            if save_pictures_in_excel:
                result.add_picture(PoseEstimator_.get_image(),time_stamp, count, PicturesamplingTime=50)
            
            plt.pause(.001)

            if cv2.waitKey(5) & 0xFF == 27:
                break


        cap.release()
        
    finally:
        result.save_excel(result.error_list)
