import numpy as np


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
        rs_flexion_FB = np.rad2deg(np.arcsin(np.dot(rightElbowLine,chest_zaxis)/(np.linalg.norm(rightElbowLine)*np.linalg.norm(chest_zaxis))))
        rs_abduction_CWCCW = np.rad2deg(np.arcsin(np.dot(rightElbowLine,-chest_xaxis)/(np.linalg.norm(rightElbowLine)*np.linalg.norm(chest_xaxis))))
        
        leftElbowLine = leftElbow - leftShoulder
        ls_flexion_FB = np.rad2deg(np.arcsin(np.dot(leftElbowLine,chest_zaxis)/(np.linalg.norm(leftElbowLine)*np.linalg.norm(chest_zaxis))))
        ls_abduction_CCWCW = np.rad2deg(np.arcsin(np.dot(leftElbowLine,chest_xaxis)/(np.linalg.norm(leftElbowLine)*np.linalg.norm(chest_xaxis))))
        
        return rs_flexion_FB, rs_abduction_CWCCW, ls_flexion_FB, ls_abduction_CCWCW
