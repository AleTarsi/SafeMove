import cv2
import mediapipe as mp
import os
import time
from sm_06_PoseEstimator import PoseEstimator
from sm_02_GUI import Gui
from sm_00_utils import computeFPS
import matplotlib.pyplot as plt
from sm_04_ResultsLogger import ResultsLogger
from sm_07_RiskAssessment import RiskAssessment
            
            
save_pictures_in_excel = False
visualizePose = True
speed_up_rate = 1 # integer value greater than 1, it set the number of frames to skip in the video btw one computation and another
fps_input_video = 30
period_btw_frames = 1/fps_input_video
count = 0 

current_folder = os.getcwd()
video_folder ="/videos/"
################# Select video here #######################
video = 'video_10'
###################################################################
path = current_folder + video_folder + video + ".mp4"

assert os.path.exists(path), f"Video does not exist in the specified path: {path}.mp4"

cap = cv2.VideoCapture(path) # 0 for webcam
# cap = cv2.VideoCapture(0) # 0 for webcam

logger = ResultsLogger(folder_path=current_folder, source_video=video)
NN = mp.solutions.holistic #mp.solutions.pose

NN.Holistic(static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

with NN.Holistic() as PoseNN: # very important for the sake of computation efficiency, not sure why though.
    try: 
        gui = Gui()
        poseEstimator = PoseEstimator()
        poseEstimator.setBaricenterLimit(0.8) # value btw 0.0 and 1.0, where 0.0 means you are on one foot any time the hip projection is not in the middle of your feet, and 1.0 you are considered on one foot only if your hip ground projection is over your feet position.
        poseEstimator.setMaxKneeDifference(20) # maximum difference in your knee angles before you are considered on one foot

        while cap.isOpened():
            for i in range(speed_up_rate):
                success, image = cap.read() 
                if success:
                    count = count + 1    

            start = time.time()
            gui.clear()
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
            
            # print(results)
            angle_list = poseEstimator.run(image, results)
            
            if angle_list != -1:
                    
                time_stamp = count*period_btw_frames
                
                logger.pose_data.loc[len(logger.pose_data.index)] = [time_stamp, *angle_list] # the Asterisk unpack tuples or lists
                # print(logger.pose_data)
                
                if visualizePose:
                    gui.draw3D(results.pose_world_landmarks)
                    gui.drawLandmark(image,results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks, NN)
                
            end = time.time()
            fps = computeFPS(end,start,speed_up_rate)
            
            gui.showText(image, f'FPS: {int(fps)}', (20,450))   
            
            cv2.imshow('Head Pose Estimation', image)
            
            if save_pictures_in_excel:
                logger.add_picture(image,time_stamp, count, PicturesamplingTime=50)
            
            plt.pause(.001)

            if cv2.waitKey(5) & 0xFF == 27:
                break


        cap.release()
        
    finally:
        reba_score, aggregated_reba_score = RiskAssessment.fromDataFrame2Reba(logger.pose_data)
        logger.save_excel(logger.pose_data, reba_score, aggregated_reba_score)
        



