import numpy as np
from sm_00_utils import normalize


class Computation:

    def BodyAxes(self, leftHip):
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
    
    def BackAxes(self, left_shoulder_point, chest):
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
    
    def BackAngles(self, body_xaxis, body_yaxis, body_zaxis, chest_xaxis, chest_yaxis, chest_zaxis):
        
        if np.cross(chest_xaxis,body_xaxis)[2]>= 0: # if the cross product points upward 
            sign_LR = 1
        else:
            sign_LR = -1
        chest_LR = sign_LR * np.rad2deg(np.arccos(np.dot(body_xaxis,chest_xaxis)/(np.linalg.norm(body_xaxis)*np.linalg.norm(chest_xaxis))))
        chest_FB = np.rad2deg(np.arccos(np.dot(body_zaxis,chest_zaxis)/(np.linalg.norm(body_zaxis)*np.linalg.norm(chest_zaxis))))
        chest_Rot = np.rad2deg(np.arcsin(np.dot(body_xaxis,chest_yaxis)/(np.linalg.norm(body_xaxis)*np.linalg.norm(chest_yaxis))))
            
        return chest_LR, chest_FB, chest_Rot

    def ShoulderAngles(self, rightShoulder, rightElbow, leftShoulder, leftElbow, chest_zaxis, chest_xaxis):
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
    
    def ElbowAngles(self, rightShoulder,rightElbow, rightWrist, leftShoulder, leftElbow, leftWrist):
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
    
    def WristAngles(self, rightElbow, rightWrist, rightHand, rightIndex, rightPinky, leftElbow, leftWrist, leftHand, leftIndex, leftPinky):
        '''
        rw: right wrist
        lw: left wrist
        Observation: Here you cannot use the arccosine wrt a line going further the writst as you can twist your wrist laterally and UD, you need to use arcsin with a line poining up your wrist
        '''
        
        '''
        1- compute the elbow line as the line connecting wrist and elbow
        2- compute the lateral direction using the line connecting the index to the pinky finger
        3- Compute the third direction as the one pointing up and starting from the wrist.
        4- Use that orthogonal direction to compute the wrist UD angle. As the line
        '''
        
        rightHandLine = normalize(rightHand - rightWrist) # line connecting the wrist and the hand centered in zero
        rightWristLine = normalize(rightWrist - rightElbow) # line connecting the elbow and the wrist centered in zero
        rightPalmLine = normalize(rightIndex - rightPinky) # line connecting the pinky and the index centered in zero
        rightOrthogonalPalmLine = normalize(np.cross(rightWristLine, rightPalmLine))
        rw_flexion_UD = np.rad2deg(np.arcsin(np.dot(rightHandLine,rightOrthogonalPalmLine)/(np.linalg.norm(rightHandLine)*np.linalg.norm(rightOrthogonalPalmLine))))
        
        leftHandLine = normalize(leftHand - leftWrist) # line connecting the wrist and the hand centered in zero
        leftWristLine = normalize(leftWrist - leftElbow) # line connecting the elbow and the shoulder centered in zero
        leftPalmLine = normalize(leftIndex - leftPinky)
        leftOrthogonalPalmLine = normalize(np.cross(leftWristLine, -leftPalmLine))
        lw_flexion_UD = np.rad2deg(np.arcsin(np.dot(leftHandLine,leftOrthogonalPalmLine)/(np.linalg.norm(leftHandLine)*np.linalg.norm(leftOrthogonalPalmLine))))
        
        return rw_flexion_UD, lw_flexion_UD, leftWristLine, leftPalmLine, leftOrthogonalPalmLine, rightWristLine, rightPalmLine, rightOrthogonalPalmLine
        
        pass

    def KneeAngles(self, rightKnee, leftKnee, rightHip, leftHip, rightAnkle, leftAnkle):
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
