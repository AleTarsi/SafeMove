# MIT License
#
# Copyright (c) [YEAR] [Your Name or Your Organization]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cv2
import mediapipe as mp
import datetime
import yaml
import numpy as np
import os
import time
from sm_02_GUI import Gui
from sm_04_ResultsLogger import ResultsLogger
from sm_06_PoseEstimator import PoseEstimator
from sm_07_RiskAssessment import RiskAssessment
import matplotlib.pyplot as plt
from pathlib import Path

WORKSPACE = os.environ.get('SAFE_MOVE_PATH', None)
assert WORKSPACE is not None, "Prepare your workspace, follow the instructions in the README.md file" 

class SafeMove():
    def __init__(self, Force, Coupling, Activity):
        self.Force = Force
        self.Coupling = Coupling
        self.Activity = Activity
        self.count = 0 # frame counter
        self.param = yaml.safe_load(open(os.path.join(WORKSPACE, "config", "config.yaml")))
        self.gui = Gui()
        self.poseEstimator = PoseEstimator()

    def start_streaming(self):
        path_video = os.path.join(WORKSPACE, "videos", self.param["video"])
        assert os.path.exists(path_video), f"Video does not exist in the specified path: {path_video}"
        if self.param['webcam']: path_video = 0 # set the path to 0 to use the webcam

        cap = cv2.VideoCapture(path_video) # open the video stream
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

        return cap, width, height

    def start_logging(self, width, height):
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        current_time = datetime.datetime.now()
        output_path = os.path.join(WORKSPACE, 'output', Path(self.param["video"]).stem , str(current_time.year) + '_' + str(current_time.month) + '_' + str(current_time.day) + '__' + str(current_time.hour) + '_' + str(current_time.minute) + '_' + str(current_time.second))
        out = cv2.VideoWriter(os.path.join(output_path, 'SafeMoveResults.mp4'), fourcc, 15.0, size)
        self.logger = ResultsLogger(folder_path=WORKSPACE, output_path=output_path)
        return out
    
    def computeFPS(self, start, end):
        frames_to_skip = self.param['speed_up']
        totalTime = end - start

        try:
            fps_output = (1 / totalTime)*frames_to_skip
        except:
            fps_output= -1
                    
        return fps_output
    
    def inference(self, PoseNN, image):
            self.gui.clear()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Flip the image horizontally for a later selfie-view display, also convert the color space from BGR to RGB
            height, width, _ = image.shape

            image.flags.writeable = False # Make the image read-only to improve performance
            
            # Get the result
            results = PoseNN.process(image)
        
            image.flags.writeable = True # Make the image writeable to draw landmarks
            
            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Transform the landmarks to angles
            angle_list = self.poseEstimator.run(image, results, self.gui)
            
            if angle_list != -1:
                period_btw_frames = 1/self.param['fps_input_video']
                time_stamp = self.count*period_btw_frames
                
                # Log the results
                self.logger.pose_data.loc[len(self.logger.pose_data.index)] = [time_stamp, *angle_list] # the Asterisk unpack tuples or lists
                
                if self.param['visualize_pose']:
                    self.gui.draw3D(results.pose_world_landmarks)
                    self.gui.drawLandmark(image,results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks, NN)
            
                self.gui.showText(image, f'time: {np.round(time_stamp, decimals=2)}', (10,height-50))   
            
                cv2.imshow('Head Pose Estimation', cv2.resize(image, (int(width*1.5), int(height*1.5))))
                
                if self.param['save_pictures_in_excel']:
                    self.logger.add_picture(image,time_stamp, self.count, PicturesamplingTime=50)

                plt.pause(.001)
    
    def get_frame(self, cap):
        for i in range(int(np.ceil(self.param['speed_up']))): # speed_up is an integer indicating the frames to skip 
            success, image = cap.read() 
        return success, image


if __name__ == "__main__":

    Force = int(input("How much force score 0,1,2 or 3? "))
    assert Force in range(0,4)
    Coupling = int(input("How is the coupling score 0,1,2 or 3? "))
    assert Coupling in range(0,4)
    Activity = int(input("How is the activity score 0,1,2 or 3? "))
    assert Activity in range(0,4)

    sm = SafeMove(Force, Coupling, Activity)
    cap, width, height = sm.start_streaming() # open the video stream
    out = sm.start_logging(width, height) # start the logger
    NN = mp.solutions.holistic #mp.solutions.pose
    
    NN.Holistic(static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    with NN.Holistic() as PoseNN:
        try:
            while cap.isOpened():
                success, image = sm.get_frame(cap)
                if success:
                    sm.count = sm.count + 1    
                    start = time.time()
                    sm.inference(PoseNN, image)
                    end = time.time()
                    fps_output_video = sm.computeFPS(end, start)
                    print(f'FPS: {fps_output_video}')
                    out.write(image)
                # Exit the loop when the ESC key is pressed or the video ends
                if cv2.waitKey(5) & 0xFF == 27 or not success:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit(0)
            
        finally: # Execute this code block at the end of the loop
            reba_score, aggregated_reba_score = RiskAssessment.fromDataFrame2Reba(sm.logger.pose_data, sm.Force, sm.Coupling, sm.Activity)
            sm.logger.save_pie_chart_angles(reba_score)
            sm.logger.save_pie_chart_bin_score(aggregated_reba_score)
            sm.logger.save_excel(sm.logger.pose_data, reba_score, aggregated_reba_score)
