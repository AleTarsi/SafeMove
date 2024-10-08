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

import numpy as np
from sm_00_utils import normalize, bcolors


class Pose2Angles:

    def BodyAxes(leftHip):
        '''
        Return three axes with magnitude 0.5 centered in zero and orientated as the Waist of the person.
        '''
        world_xaxis = np.array([0.5,0,0])
        world_yaxis = np.array([0,0,0.5])
        world_zaxis = np.array([0,-0.5,0])

        left_hip_line = leftHip
        left_hip_line[2] = 0 # We set to zero the component pointing up (world_yaxis),
        body_xaxis = 0.5 * (left_hip_line)/np.linalg.norm(left_hip_line)
        body_yaxis = world_yaxis

        z_dir = np.cross(left_hip_line,world_yaxis)
        body_zaxis = 0.5 * (z_dir)/np.linalg.norm(z_dir)

        return body_xaxis, body_yaxis, body_zaxis
    
    def BackAxes(left_shoulder_point, chest):
        '''
        Return three axes with magnitude 0.5 centered in zero and orientated as the chest of the person.
        '''
        world_xaxis = np.array([0.5,0,0])
        world_yaxis = np.array([0,0,0.5])
        world_zaxis = np.array([0,-0.5,0])
        origin = np.array([0,0,0])
        
        left_shoulder_point[2] = chest[2] # We set to the chest level the component pointing up (world_yaxis)
        shoulder_xaxis = 0.5 * (left_shoulder_point-chest)/np.linalg.norm(left_shoulder_point-chest)
        shoulder_yaxis = 0.5 * (chest - origin)/np.linalg.norm(chest - origin)
        z_dir = np.cross(shoulder_xaxis,shoulder_yaxis)
        shoulder_zaxis = 0.5 * (z_dir)/np.linalg.norm(z_dir)
        
        return shoulder_xaxis, shoulder_yaxis, shoulder_zaxis
    
    def BackAngles(body_xaxis, body_yaxis, body_zaxis, chest_xaxis, chest_yaxis, chest_zaxis):
        
        if np.cross(chest_xaxis,body_xaxis)[2]>= 0: # if the cross product points upward 
            sign_LR = 1
        else:
            sign_LR = -1
        chest_LR = sign_LR * np.rad2deg(np.arccos(np.dot(body_xaxis,chest_xaxis)/(np.linalg.norm(body_xaxis)*np.linalg.norm(chest_xaxis))))
        chest_FB = np.rad2deg(np.arccos(np.dot(body_zaxis,chest_zaxis)/(np.linalg.norm(body_zaxis)*np.linalg.norm(chest_zaxis))))
        chest_Rot = np.rad2deg(np.arcsin(np.dot(body_xaxis,chest_yaxis)/(np.linalg.norm(body_xaxis)*np.linalg.norm(chest_yaxis))))
            
        return chest_LR, chest_FB, chest_Rot

    def ShoulderAngles(rightShoulder, rightElbow, leftShoulder, leftElbow, chest_zaxis, chest_xaxis):
        '''
        rs: right shoulder
        ls: left shoulder
        '''
        rightElbowLine = rightElbow - rightShoulder # line connecting the elbow and the shoulder centered in zero, remember that also chest_*axis is centered in zero
        # chest_zaxis points forward, i.e. the direction of the forehead
        rs_flexion_FB = np.rad2deg(np.arcsin(np.dot(rightElbowLine,chest_zaxis)/(np.linalg.norm(rightElbowLine)*np.linalg.norm(chest_zaxis))))
        rs_abduction_CWCCW = np.rad2deg(np.arcsin(np.dot(rightElbowLine,-chest_xaxis)/(np.linalg.norm(rightElbowLine)*np.linalg.norm(chest_xaxis))))
        
        leftElbowLine = leftElbow - leftShoulder
        # chest_zaxis points forward, i.e. the direction of the forehead
        ls_flexion_FB = np.rad2deg(np.arcsin(np.dot(leftElbowLine,chest_zaxis)/(np.linalg.norm(leftElbowLine)*np.linalg.norm(chest_zaxis))))
        ls_abduction_CCWCW = np.rad2deg(np.arcsin(np.dot(leftElbowLine,chest_xaxis)/(np.linalg.norm(leftElbowLine)*np.linalg.norm(chest_xaxis))))
        
        return rs_flexion_FB, rs_abduction_CWCCW, ls_flexion_FB, ls_abduction_CCWCW
    
    def ElbowAngles(rightShoulder,rightElbow, rightWrist, leftShoulder, leftElbow, leftWrist):
        '''
        re: right elbow
        le: left elbow
        '''
        rightWristLine = rightWrist - rightElbow # line connecting the elbow and the wrist centered in zero
        rightElbowLine = rightElbow - rightShoulder # line connecting the shoulder and the elbow centered in zero
        re_flexion = np.rad2deg(np.arccos(np.dot(rightWristLine,rightElbowLine)/(np.linalg.norm(rightWristLine)*np.linalg.norm(rightElbowLine))))
        
        leftWristLine = leftWrist - leftElbow # line connecting the elbow and the wrist centered in zero
        leftElbowLine = leftElbow - leftShoulder # line connecting the elbow and the shoulder centered in zero
        le_flexion = np.rad2deg(np.arccos(np.dot(leftWristLine,leftElbowLine)/(np.linalg.norm(leftWristLine)*np.linalg.norm(leftElbowLine))))
        
        return re_flexion, le_flexion
    
    def WristAngles(Elbow, Wrist, Hand, Index, Pinky):
        '''
        lw: left wrist
        le: left elbow
        Observation: Here you cannot use the arccosine wrt a line going further the writst as you can twist your wrist laterally and UD, you need to use arcsin with a line poining up your wrist
        '''
        
        '''
        1- compute the elbow line as the line connecting wrist and elbow
        2- compute the lateral direction using the line connecting the index to the pinky finger
        3- Compute the third direction as the one pointing up and starting from the wrist.
        4- Use that orthogonal direction to compute the wrist UD angle. As the line
        '''
        
        up_direction = np.array([0,0,1.0]) # Line pointing up in the world reference frame, it is used to compute the 
        HandLine = normalize(Hand - Wrist) # line connecting the wrist and the hand centered in zero
        WristLine = normalize(Wrist - Elbow) # line connecting the elbow and the wrist centered in zero
        PalmLine = normalize(Index - Pinky) # line connecting the pinky and the index centered in zero
        OrthogonalPalmLine = normalize(np.cross(WristLine, PalmLine))
        wrist_flexion_UD = np.rad2deg(np.arcsin(np.dot(HandLine,OrthogonalPalmLine)/(np.linalg.norm(HandLine)*np.linalg.norm(OrthogonalPalmLine))))
        elbow_rotation_PS = np.rad2deg(-np.arcsin(np.dot(PalmLine,up_direction)/(np.linalg.norm(PalmLine)))) # there is a minus because a rotation toward the body must be positive
        wrist_rotation_UR = np.rad2deg(np.arccos(np.dot(PalmLine,WristLine)/(np.linalg.norm(PalmLine)*np.linalg.norm(WristLine))))
        
        return wrist_flexion_UD, elbow_rotation_PS, wrist_rotation_UR, WristLine, PalmLine, OrthogonalPalmLine

    def KneeAngles(rightKnee, leftKnee, rightHip, leftHip, rightAnkle, leftAnkle):
        '''
        rk: right knee
        lk: left knee
        
        1- compute the knee line as the line connecting the hip and the knee
        2- compute the feet line as the line connecting the knee and the foot
        3- compute the angle in btw this two lines
        '''
        leftkneeLine = leftHip - leftKnee
        leftFeetLine = leftKnee - leftAnkle
        lk_flexion = np.rad2deg(np.arccos(np.dot(leftkneeLine,leftFeetLine)/(np.linalg.norm(leftkneeLine)*np.linalg.norm(leftFeetLine))))
        
        rightkneeLine = rightHip - rightKnee
        rightFeetLine = rightKnee - rightAnkle
        rk_flexion = np.rad2deg(np.arccos(np.dot(rightkneeLine,rightFeetLine)/(np.linalg.norm(rightkneeLine)*np.linalg.norm(rightFeetLine))))
        
        return rk_flexion, lk_flexion
    
    def ComputeContactPoints(rk_flexion, lk_flexion, max_knee_difference, rightAnkle, leftAnkle, Hip, min_baricenter_position, max_baricenter_position):
        ############################### Computation of number of Contact points ###########################################
                
                '''
                1- Verify whether the knee angles' difference is over a certain threshold as first check
                2- Verify whether the lateral position of the pelvis are outside a certain threshold as second check
                3- From these two checks determine the number of contact points
                '''
                
                contact_points = 2
                
                #Point 1
                knee_difference = abs(rk_flexion - lk_flexion)
                if knee_difference >= max_knee_difference:
                    contact_points = 1
                    
                '''
                Point 2 
                - it'd be sufficient to compute the position of the central point of the pelvis wrt the line connecting the feet 
                - Project the Hip point onto the line connecting the feet, it should be 0.5 the value when the pelvis is in centered 
                
                Idea: compute the projection line as described in https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line
                '''
                #Point 2
                InitialPoint = rightAnkle
                FinalPoint = leftAnkle
                FinalPoint[2] = InitialPoint[2] # We set the y-cordinate to the same. Look in trello for the contact point computation theory
                foot2footLine = FinalPoint - InitialPoint
                Origin2footLine = Hip - InitialPoint # Hip seat on the origin
                ProjectionLine = np.dot(Origin2footLine,foot2footLine)/np.dot(foot2footLine,foot2footLine)*foot2footLine
                
                # This value should be btw 1 or -1
                baricenterDirection = np.dot(ProjectionLine,foot2footLine)/(np.linalg.norm(ProjectionLine)*np.linalg.norm(foot2footLine))
                # print(baricenterDirection)
                
                # signed length ration between ProjectionLine and foot2footLine
                baricenterValue = np.linalg.norm(ProjectionLine)/np.linalg.norm(foot2footLine)*baricenterDirection
                
                if baricenterValue < min_baricenter_position or baricenterValue > max_baricenter_position:
                    contact_points = 1
                    
                return contact_points, knee_difference
