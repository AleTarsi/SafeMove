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
import sys
import os

current_folder = os.getcwd()

################# Select video here #######################
video = 'video_10'
time = '2024_6_7__11_43_11'
###################################################################
video_folder ="\output\\"
path = current_folder + video_folder + video + '\\'+ time +'\\'+ "00_SafeMoveResults.mp4"


# load input video
cap = cv2.VideoCapture(path)
if (cap.isOpened() == False):
    print("!!! Failed cap.isOpened()")
    print(f'The {path} does not exist!')
    sys.exit(-1)

# retrieve the total number of frames
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames2skip = 2

# loop to read every frame of the video
while (cap.isOpened()):

    # capture a frame
    ret, frame = cap.read()
    if ret == False:
        print("!!! Failed cap.read()")
        break

    cv2.imshow('video', frame)

    # check if 'p' was pressed and wait for a 'b' press
    key = cv2.waitKey(5)
    # Press space bar to pause the video
    if (key & 0xFF == 32):

        # sleep here until a valid key is pressed
        while (True):
            key = cv2.waitKey(5)

            # check if space is pressed and resume playing
            if (key & 0xFF == 32):
                frames2skip = 2
                break

            # check if 'b' is pressed and rewind video 
            if (key & 0xFF == ord('b') ):
                cur_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print('* At frame #' + str(cur_frame_number))

                prev_frame = cur_frame_number
                if (cur_frame_number > 1):
                    prev_frame -= frames2skip
                else:
                    print('* Rewind to frame #0')

                print('* Rewind to frame #' + str(prev_frame))
                cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
                ret, frame = cap.read()
                cv2.imshow('video', frame)
            
            # check if 'f' is pressed and forward the video 
            if (key & 0xFF == ord('f')):
                cur_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print('* At frame #' + str(cur_frame_number))

                next_frame = cur_frame_number
                if (cur_frame_number > 1):
                    next_frame += frames2skip
                else:
                    print('* Rewind to frame #0')

                print('* Rewind to frame #' + str(next_frame))
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                ret, frame = cap.read()
                cv2.imshow('video', frame)
            
            if (key & 0xFF == ord('u') ):
                if frames2skip < 10:
                    frames2skip += 1
                else: 
                    frames2skip = 10
            
            if (key & 0xFF == ord('d') ):
                if frames2skip > 2:
                    frames2skip -= 1
                else: 
                    frames2skip = 2

            # check if 'r' is pressed and rewind video to frame 0, then resume playing
            if (key & 0xFF == ord('r')):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frames2skip = 2
                break

    # exit when 'q' is pressed to quit
    elif (key & 0xFF == ord('q')):
        break

# release resources
cap.release()
cv2.destroyAllWindows()
