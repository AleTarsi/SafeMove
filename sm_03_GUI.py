import matplotlib.pyplot as plt
import numpy as np

class Gui:
    def __init__(self):
        #Array of points
        self.tracked_points = []
        #Graphical elements
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),projection='3d')
        self.ax.grid(visible=True, which='both')
        plt.ion()
        plt.pause(.05)
        
    def parsingBodyPoints(self, landmarks):
        self.tracked_points = landmarks
        self.draw()
        
    def draw(self):
        print("hello")
        pass
    
