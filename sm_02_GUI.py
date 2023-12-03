import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from sm_00_utils import computeMidPosition

class Gui:
    def __init__(self):


        #Set up graphical elements
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),projection='3d')
                
        self.ax.grid(visible=True, which='both')
        
        

    def draw3D(self,landmarks, extraPoint=False):

        landmark_point = []

        for index, landmark in enumerate(landmarks.landmark):
            landmark_point.append([landmark.visibility, (landmark.x, landmark.y, landmark.z)])
            
        self.ax.cla()
        
        self.ax.plot([0,0.5], [0,0],zs=[0,0], color="red")
        self.ax.plot([0,0], [0,0],zs=[0,0.5], color="green")
        self.ax.plot([0,0], [0,-0.5],zs=[0,0], color="blue")
        
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(-1, 1)


        # face
        face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        face_x, face_y, face_z = [], [], []
        for index in face_index_list:
            point = landmark_point[index][1]
            face_x.append(point[0])
            face_y.append(point[2])
            face_z.append(point[1] * (-1))
        self.ax.scatter(face_x, face_y, face_z)
        
        # right arm
        right_arm_index_list = [11, 13, 15, 17, 19, 21]
        right_arm_x, right_arm_y, right_arm_z = [], [], []
        for index in right_arm_index_list:
            point = landmark_point[index][1]
            right_arm_x.append(point[0])
            right_arm_y.append(point[2])
            right_arm_z.append(point[1] * (-1))
        self.ax.plot(right_arm_x, right_arm_y, right_arm_z)

        # left arm
        left_arm_index_list = [12, 14, 16, 18, 20, 22]
        left_arm_x, left_arm_y, left_arm_z = [], [], []
        for index in left_arm_index_list:
            point = landmark_point[index][1]
            left_arm_x.append(point[0])
            left_arm_y.append(point[2])
            left_arm_z.append(point[1] * (-1))
        self.ax.plot(left_arm_x, left_arm_y, left_arm_z)

        # right body
        right_body_side_index_list = [11, 23, 25, 27, 29, 31]
        right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
        for index in right_body_side_index_list:
            point = landmark_point[index][1]
            right_body_side_x.append(point[0])
            right_body_side_y.append(point[2])
            right_body_side_z.append(point[1] * (-1))
        self.ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)

        # left body
        left_body_side_index_list = [12, 24, 26, 28, 30, 32]
        left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
        for index in left_body_side_index_list:
            point = landmark_point[index][1]
            left_body_side_x.append(point[0])
            left_body_side_y.append(point[2])
            left_body_side_z.append(point[1] * (-1))
        self.ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)

        # shoulder
        shoulder_index_list = [11, 12]
        shoulder_x, shoulder_y, shoulder_z = [], [], []
        for index in shoulder_index_list:
            point = landmark_point[index][1]
            shoulder_x.append(point[0])
            shoulder_y.append(point[2])
            shoulder_z.append(point[1] * (-1))
        self.ax.plot(shoulder_x, shoulder_y, shoulder_z)

        # waist
        waist_index_list = [23, 24]
        waist_x, waist_y, waist_z = [], [], []
        for index in waist_index_list:
            point = landmark_point[index][1]
            waist_x.append(point[0])
            waist_y.append(point[2])
            waist_z.append(point[1] * (-1))
        self.ax.plot(waist_x, waist_y, waist_z)
        
    
        if extraPoint is not False:
            self.ax.plot(np.linspace(face_x[0],extraPoint[0],10),np.linspace(face_y[0],extraPoint[1],10),np.linspace(face_z[0],extraPoint[2],10))
            
        plt.pause(.001)

        return
    
    def DrawTrunk(self,trunk_point):
        '''
        Pass points after the conversion [x,y,z] becomes [x,z,-y]
        '''
        self.ax.cla()
        
        self.ax.plot([0,0.5], [0,0],zs=[0,0], color="red")
        self.ax.plot([0,0], [0,0],zs=[0,0.5], color="green")
        self.ax.plot([0,0], [0,-0.5],zs=[0,0], color="blue")
        
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(-1, 1)
        
        colors_list = list(mcolors.TABLEAU_COLORS)       
        # trunk
        trunk_x, trunk_y, trunk_z, color = [], [], [], []
        for index, point in enumerate(trunk_point):
            trunk_x.append(point[0])
            trunk_y.append(point[1])
            trunk_z.append(point[2])
            color.append(colors_list[index])
        self.ax.scatter(trunk_x, trunk_y, trunk_z, c=color)
        
    def BodyReferenceFrame(self, left_hip_line):
        '''
        We define a reference frame fixed to the hip and rotating based on the body orientation
        '''
        world_xaxis = np.array([0.5,0,0])
        world_yaxis = np.array([0,0,0.5])
        world_zaxis = np.array([0,-0.5,0])
        
        left_hip_line[2] = 0 # We set to zero the component pointing up (world_yaxis)
        body_xaxis = 0.5 * (left_hip_line)/np.linalg.norm(left_hip_line)
        self.ax.plot([0,body_xaxis[0]], [0,body_xaxis[1]],zs=[0,body_xaxis[2]], color="red")
        
        body_yaxis = world_yaxis
        if 0: # not really used as it is coincident with the world reference frame
            self.ax.plot([0,body_yaxis[0]], [0,body_yaxis[1]],zs=[0,body_yaxis[2]], color="green")
        
        body_zaxis = np.cross(left_hip_line,world_yaxis)
        z_dir = 0.5 * (body_zaxis)/np.linalg.norm(body_zaxis)
        self.ax.plot([0,z_dir[0]], [0,z_dir[1]],zs=[0,z_dir[2]], color="blue")
        # plt.pause(.001)
        
        return body_xaxis, body_yaxis, body_zaxis
    
    def ChestReferenceFrame(self, left_shoulder_point, chest):
        '''
        We define a reference frame fixed to the hip and rotating based on the body orientation
        '''
        world_xaxis = np.array([0.5,0,0])
        world_yaxis = np.array([0,0,0.5])
        world_zaxis = np.array([0,-0.5,0])
        origin = np.array([0,0,0])
        
        # left_shoulder_point[2] = chest[2] # We set to the chest level the component pointing up (world_yaxis)
        shoulder_xaxis = 0.5 * (left_shoulder_point-chest)/np.linalg.norm(left_shoulder_point-chest)
        self.ax.plot([chest[0],shoulder_xaxis[0]+chest[0]], [chest[1],shoulder_xaxis[1]+chest[1]],zs=[chest[2],shoulder_xaxis[2]+chest[2]], color="red")
        
        shoulder_yaxis = 0.5 * (chest - origin)/np.linalg.norm(chest - origin)
        
        self.ax.plot([chest[0],shoulder_yaxis[0]+chest[0]], [chest[1],shoulder_yaxis[1]+chest[1]],zs=[chest[2],shoulder_yaxis[2]+chest[2]], color="green")
        
        z_dir = np.cross(shoulder_xaxis,shoulder_yaxis)
        shoulder_zaxis = 0.5 * (z_dir)/np.linalg.norm(z_dir)
        self.ax.plot([chest[0],shoulder_zaxis[0]+chest[0]], [chest[1],shoulder_zaxis[1]+chest[1]],zs=[chest[2],shoulder_zaxis[2]+chest[2]], color="blue")
        plt.pause(.001)
        
        return shoulder_xaxis, shoulder_yaxis, shoulder_zaxis

        
    def Draw3DFace(self,face_point):
        '''
        Pass points after the conversion [x,y,z] becomes [x,z,-y]
        '''
        self.ax.cla()
        
        self.ax.set_xlim3d(-0.5, 0.5)
        self.ax.set_ylim3d(-1, -0.5)
        self.ax.set_zlim3d(0, 1)
        
        colors_list = list(mcolors.TABLEAU_COLORS)       
        # face
        face_x, face_y, face_z, color = [], [], [], []
        for index, point in enumerate(face_point):
            face_x.append(point[0])
            face_y.append(point[1])
            face_z.append(point[2])
            color.append(colors_list[index])
        self.ax.scatter(face_x, face_y, face_z, c=colors_list[index])
            
        plt.pause(.001)
         
    
    def drawRotationVector(self,vect):
        self.ax2.cla()
        self.ax2.set_xlim3d(-0.5, 0.5)
        self.ax2.set_ylim3d(-0.5,0.5)
        self.ax2.set_zlim3d(-0.5, 0.5)
        self.ax2.plot(np.linspace(0,vect[0],10), np.linspace(0,vect[1],10), np.linspace(0,vect[2],10), c='red')
        
        plt.pause(.001)

