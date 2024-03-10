import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import mediapipe as mp
import numpy as np
import cv2
from sm_00_utils import computeMidPosition

class Gui:
    def __init__(self):

        #Set up graphical elements
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),projection='3d')
                
        self.ax.grid(visible=True, which='both')
        
    def drawLandmark(self, image, pose_landmarks, NN):
        mp_drawing = mp.solutions.drawing_utils
        # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Render detections
        mp_drawing.draw_landmarks(image, pose_landmarks, NN.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )

    def showText(self, image, txt, pose): # f'FPS: {int(fps)}'
        '''
        Note that pose argument must be a tuple
        '''
        cv2.putText(image, txt, pose, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    
    def draw3D(self,landmarks, extraPoint=False):

        landmark_point = []

        for index, landmark in enumerate(landmarks.landmark):
            landmark_point.append([landmark.visibility, (landmark.x, landmark.y, landmark.z)])
            
        
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
            
        return
    
    def DrawTrunk(self,trunk_point):
        '''
        Pass points after the conversion [x,y,z] becomes [x,z,-y]
        '''
        
        self.ax.plot([0,0.5], [0,0],zs=[0,0], color="red")
        self.ax.plot([0,0], [0,0],zs=[0,0.5], color="green")
        self.ax.plot([0,0], [0,-0.5],zs=[0,0], color="blue")
        
        colors_list = list(mcolors.TABLEAU_COLORS)       
        # trunk
        trunk_x, trunk_y, trunk_z, color = [], [], [], []
        for index, point in enumerate(trunk_point):
            trunk_x.append(point[0])
            trunk_y.append(point[1])
            trunk_z.append(point[2])
            color.append(colors_list[index])
        self.ax.scatter(trunk_x, trunk_y, trunk_z, c=color)
        
    def BodyReferenceFrame(self, body_xaxis, body_yaxis, body_zaxis):
        '''
        We plot a reference frame fixed to the hip and rotating based on the body orientation
        '''
              
        self.ax.plot([0,body_xaxis[0]], [0,body_xaxis[1]],zs=[0,body_xaxis[2]], color="red")
        
        if 0: # not really used as it is coincident with the world reference frame
            self.ax.plot([0,body_yaxis[0]], [0,body_yaxis[1]],zs=[0,body_yaxis[2]], color="green")
        
        self.ax.plot([0,body_zaxis[0]], [0,body_zaxis[1]],zs=[0,body_zaxis[2]], color="blue")
        # plt.pause(.001)
        
    
    def ChestReferenceFrame(self, chest_xaxis, chest_yaxis, chest_zaxis, chest):
        '''
        We define a reference frame fixed to the hip and rotating based on the body orientation
        '''
        
        self.ax.plot([chest[0],chest_xaxis[0]+chest[0]], [chest[1],chest_xaxis[1]+chest[1]],zs=[chest[2],chest_xaxis[2]+chest[2]], color="red")
        
        self.ax.plot([chest[0],chest_yaxis[0]+chest[0]], [chest[1],chest_yaxis[1]+chest[1]],zs=[chest[2],chest_yaxis[2]+chest[2]], color="green")
        
        self.ax.plot([chest[0],chest_zaxis[0]+chest[0]], [chest[1],chest_zaxis[1]+chest[1]],zs=[chest[2],chest_zaxis[2]+chest[2]], color="blue")
    
    def RshoulderAxes(self, chest_xaxis, chest_yaxis, chest_zaxis, rightShoulder):
        self.ax.plot([rightShoulder[0],chest_xaxis[0]+rightShoulder[0]], [rightShoulder[1],chest_xaxis[1]+rightShoulder[1]],zs=[rightShoulder[2],chest_xaxis[2]+rightShoulder[2]], color="red")
        
        self.ax.plot([rightShoulder[0],chest_yaxis[0]+rightShoulder[0]], [rightShoulder[1],chest_yaxis[1]+rightShoulder[1]],zs=[rightShoulder[2],chest_yaxis[2]+rightShoulder[2]], color="green")
        
        self.ax.plot([rightShoulder[0],chest_zaxis[0]+rightShoulder[0]], [rightShoulder[1],chest_zaxis[1]+rightShoulder[1]],zs=[rightShoulder[2],chest_zaxis[2]+rightShoulder[2]], color="blue")
        
    def LshoulderAxes(self, chest_xaxis, chest_yaxis, chest_zaxis, leftShoulder):
        self.ax.plot([leftShoulder[0],chest_xaxis[0]+leftShoulder[0]], [leftShoulder[1],chest_xaxis[1]+leftShoulder[1]],zs=[leftShoulder[2],chest_xaxis[2]+leftShoulder[2]], color="red")
        
        self.ax.plot([leftShoulder[0],chest_yaxis[0]+leftShoulder[0]], [leftShoulder[1],chest_yaxis[1]+leftShoulder[1]],zs=[leftShoulder[2],chest_yaxis[2]+leftShoulder[2]], color="green")
        
        self.ax.plot([leftShoulder[0],chest_zaxis[0]+leftShoulder[0]], [leftShoulder[1],chest_zaxis[1]+leftShoulder[1]],zs=[leftShoulder[2],chest_zaxis[2]+leftShoulder[2]], color="blue")
        
        
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
        
    '''
    Those three functions can be compacted into one as they do essentially the same
    '''
        
    def DrawElbowLine(self,rightShoulder,rightElbow,leftShoulder,leftElbow):
        self.ax.plot([rightShoulder[0],rightElbow[0]], [rightShoulder[1],rightElbow[1]],zs=[rightShoulder[2],rightElbow[2]], color="orange")
        self.ax.plot([leftShoulder[0],leftElbow[0]], [leftShoulder[1],leftElbow[1]],zs=[leftShoulder[2],leftElbow[2]], color="orange")
        
    def DrawWristLine(self, rightWrist,rightElbow,leftWrist,leftElbow):
        self.ax.plot([rightElbow[0], rightWrist[0]], [rightElbow[1], rightWrist[1]],zs=[rightElbow[2], rightWrist[2]], color="purple")
        self.ax.plot([leftElbow[0], leftWrist[0]], [leftElbow[1], leftWrist[1]],zs=[leftElbow[2], leftWrist[2]], color="purple")
        
    def DrawHandLine(self, rightWrist,rightHand,leftWrist,leftHand):
        self.ax.plot([rightWrist[0], rightHand[0]], [rightWrist[1], rightHand[1]],zs=[rightWrist[2], rightHand[2]], color="red")
        self.ax.plot([leftWrist[0],leftHand[0]], [leftWrist[1],leftHand[1]],zs=[leftWrist[2],leftHand[2]], color="red")
    
    def DrawHandaxes(self, origin, x_axis, y_axis, z_axis):
        self.ax.plot([origin[0],x_axis[0]+origin[0]], [origin[1],x_axis[1]+origin[1]],zs=[origin[2],x_axis[2]+origin[2]], color="red")
        self.ax.plot([origin[0],y_axis[0]+origin[0]], [origin[1],y_axis[1]+origin[1]],zs=[origin[2],y_axis[2]+origin[2]], color="green")
        self.ax.plot([origin[0],z_axis[0]+origin[0]], [origin[1],z_axis[1]+origin[1]],zs=[origin[2],z_axis[2]+origin[2]], color="blue")

    def DrawKneeLine(self, rightKnee, leftKnee, rightHip, leftHip):
        self.ax.plot([rightHip[0], rightKnee[0]], [rightHip[1], rightKnee[1]],zs=[rightHip[2], rightKnee[2]], color="green")
        self.ax.plot([leftHip[0], leftKnee[0]], [leftHip[1], leftKnee[1]],zs=[leftHip[2], leftKnee[2]], color="green")
        
    def DrawFootLine(self, rightKnee, leftKnee, rightAnkle, leftAnkle):
        self.ax.plot([rightKnee[0], rightAnkle[0]], [rightKnee[1], rightAnkle[1]],zs=[rightKnee[2], rightAnkle[2]], color="blue")
        self.ax.plot([leftKnee[0], leftAnkle[0]], [leftKnee[1], leftAnkle[1]],zs=[leftKnee[2], leftAnkle[2]], color="blue")

    def DrawBaricenterLine(self, rightAnkle, leftAnkle, Hip):
        InitialPoint = rightAnkle
        FinalPoint = leftAnkle
        FinalPoint[2] = InitialPoint[2] # We set the y-cordinate to the same. Look in trello for the contact point computation theory
        foot2footLine = FinalPoint - InitialPoint
        Origin2footLine = Hip - InitialPoint # Hip seat on the origin
        ProjectionLine = np.dot(Origin2footLine,foot2footLine)/np.dot(foot2footLine,foot2footLine)*foot2footLine

        # print(np.dot(Origin2footLine,foot2footLine))
        self.ax.plot([InitialPoint[0], FinalPoint[0]], [InitialPoint[1], FinalPoint[1]],zs=[InitialPoint[2], FinalPoint[2]], color="orange")
        self.ax.plot([InitialPoint[0], Origin2footLine[0]], [InitialPoint[1], Origin2footLine[1]],zs=[InitialPoint[2], Origin2footLine[2]], color="black")
        self.ax.plot([InitialPoint[0], InitialPoint[0] + ProjectionLine[0]], [InitialPoint[1], InitialPoint[1] + ProjectionLine[1]],zs=[InitialPoint[2], InitialPoint[2] + ProjectionLine[2]], color="green")

        self.ax.plot([0, foot2footLine[0]], [0, foot2footLine[1]],zs=[0, foot2footLine[2]], color="purple")
        self.ax.plot([0, Origin2footLine[0]], [0, Origin2footLine[1]],zs=[0, Origin2footLine[2]], color="brown")
