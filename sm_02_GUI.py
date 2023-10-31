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
        self.ax.plot([0,0], [0,0.5],zs=[0,0], color="green")
        self.ax.plot([0,0], [0,0],zs=[0,0.5], color="blue")
        
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

        
    def Draw3DFace(self,face_point):
        
        self.ax.cla()
        
        colors_list = list(mcolors.TABLEAU_COLORS)       
        # face
        face_x, face_y, face_z = [], [], []
        for index, point in enumerate(face_point):
            face_x.append(point[0])
            face_y.append(point[1])
            face_z.append(point[2])
            self.ax.scatter(point[0], point[1], point[2], c=colors_list[index])
            
        plt.pause(.001)
         
    
    def drawRotationVector(self,vect):
        self.ax2.cla()
        self.ax2.set_xlim3d(-0.5, 0.5)
        self.ax2.set_ylim3d(-0.5,0.5)
        self.ax2.set_zlim3d(-0.5, 0.5)
        self.ax2.plot(np.linspace(0,vect[0],10), np.linspace(0,vect[1],10), np.linspace(0,vect[2],10), c='red')
        
        plt.pause(.001)

