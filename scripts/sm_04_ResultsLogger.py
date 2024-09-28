#!/usr/bin/python

''' 
This file contains the API to export the data to an excel file
'''
import pandas as pd
import cv2
import os
import datetime
import shutil
from sm_00_utils import bcolors, from_image_name_2_excel_row_value
import glob
import numpy as np
from sm_10_reba_tables import bin_score_per_articulation
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

class ResultsLogger:
  
  def __init__(self, folder_path, output_path):
    self.folder_path = folder_path
    self.output_path = output_path + '/'
    
    # Create folder if it does not exists
    os.makedirs(output_path, exist_ok=True)
      
    self.reba_image_path = os.path.join(output_path, '00_Reba.png')
      
    self.excel_name = output_path + '/00_SafeMoveResults.xlsx'
    
    self.writer = pd.ExcelWriter(self.excel_name , engine='xlsxwriter')
  
  '''
  We followed the structure and the order present in the excel
  For what regards the Data Frame the first letter indicates the positive direction and the second the negative direction
  '''
  pose_data = pd.DataFrame({'t [sec]': [],
                             'head.rotation.LR': [],
                             'head.flexion.DU': [],
                             'head.flexion.CCWCW': [],
                             'trunk.rotation.LR': [],
                             'trunk.flexion.FB': [],
                             'trunk.flexion.LR': [],
                             'R.shoulder.flexion.FB': [],
                             'R.shoulder.abduction.CWCCW': [],
                             'L.shoulder.flexion.FB': [],
                             'L.shoulder.abduction.CWCCW': [],
                             'R.elbow.flexion.UD': [],
                             'R.elbow.rotation.PS': [],
                             'L.elbow.flexion.UD': [],
                             'L.elbow.rotation.PS': [],
                             'R.wrist.flexion.UD': [],
                             'R.wrist.rotation.UR': [], 
                             'L.wrist.flexion.UD': [],
                             'L.wrist.rotation.UR': [],
                             'R.knee.flexion.UD': [],
                             'L.knee.flexion.UD': [],
                             'contact points [#]': [],
                            })
  
  def add_picture(self, img, time_stamp, count, PicturesamplingTime=50):
    '''
    In order to do not keep a struct containing all the pictures we save them in a tmp folder, and then we will delete all of them
    '''
    if count == 0 or count%PicturesamplingTime == 0:
      # img_name = "time_stamp_" + str(time_stamp) + "count_" + str(count) + ".png"
      img_name =str(count) + ".png"
      tmp_path = self.folder_path + 'tmp/' 
      
      # Create folder if it does not exists
      os.makedirs( self.folder_path + 'tmp/', exist_ok=True)

      # write_pictures
      cv2.imwrite(tmp_path + img_name, img) 
  
  def save_reba_score(self, aggregated_reba_score):
    image = cv2.imread('config/empty_reba.png')
    
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.neck.Tot'])), (385, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.trunk.Tot'])), (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.leg.Tot'])), (400, 590), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.TableA.mean'])), (400, 705), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.Force'])), (400, 790), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.TableA.Tot'])), (400, 860), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.shoulder.Tot'])), (1270, 345), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.elbow.Tot'])), (1270, 455), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.wrist.Tot'])), (1270, 570), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.TableB.mean'])), (1270, 715), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.Coupling'])), (1270, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.TableB.Tot'])), (1270, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.TableC'])), (525, 960), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.Activity'])), (650, 960), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(image, str(round(aggregated_reba_score.loc[1, 'score.REBA'])), (780, 960), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    # save image
    cv2.imwrite(self.reba_image_path, image)
    
    # create empty sheet page in the excel
    data = {}
    df = pd.DataFrame(data)
    df.to_excel(self.writer, sheet_name='Base')
    ws_reba_table = self.writer.sheets['Base']
    ws_reba_table.set_column("A:A", 2000) # set column width
    # dump image in the excel
    ws_reba_table.insert_image("A1", self.reba_image_path, {"x_scale": 1.0, "y_scale": 1.0})
    
  def save_pie_chart_angles(self, reba_score):
    parts_list = list(reba_score.keys()[1:])
    score_dict = {}
    
    for part in parts_list:
      unique, counts = np.unique(reba_score[part], return_counts=True) 
      score_dict[part] = dict(zip(unique, counts)) # it returns a dictionary, e.g 'score.head.rotation.LR' : {0: count, 1: count}
      
      mylabels = []
      myexplode = []
      bins = []
      mycolors = []
      
      
      if 'green' in bin_score_per_articulation()[part]: # if 'green' in {'green': [0, '[-45°, 45°]'], 'red': [1, '>45°, <-45°']}nt of the times the score has green value
        green_bin = bin_score_per_articulation()[part]['green'] # green_bin = [0, '[-45°, 45°]']
        
        green_score = green_bin[0] # green_score = 0
        
        # Consider the following case: # score_dict = {'score.head.rotation.LR': {1: 8}} 
        # We dont have detected cases in which the score was green, we dont have the key 0 in the dictionary, hence we don't append it to the bins
        if green_score in score_dict[part]: 

          bins.append(score_dict[part][green_score])  # score_dict[part][green_score] containts the count of the green score
          
          mylabels.append(green_bin[1]) # mylabels = '[-45°, 45°]'
          
          myexplode.append(0)
          mycolors.append('tab:green')
          
      if 'yellow' in bin_score_per_articulation()[part]:
        yellow_bin = bin_score_per_articulation()[part]['yellow']
        
        yellow_score = yellow_bin[0]
        
        if yellow_score in score_dict[part]:
          bins.append(score_dict[part][yellow_score])
          mylabels.append(yellow_bin[1])
          myexplode.append(0.1)
          mycolors.append('tab:orange')
          
      if 'red' in bin_score_per_articulation()[part]:
        red_bin = bin_score_per_articulation()[part]['red']
        
        red_score = red_bin[0]
        
        if red_score in score_dict[part]:
          bins.append(score_dict[part][red_score])
          mylabels.append(red_bin[1])
          myexplode.append(0.2)
          mycolors.append('tab:red')
          
      if 'purple' in bin_score_per_articulation()[part]:
        purple_bin = bin_score_per_articulation()[part]['purple']
        
        purple_score = purple_bin[0]
        
        if purple_score in score_dict[part]:
          bins.append(score_dict[part][purple_score])
          mylabels.append(purple_bin[1])
          myexplode.append(0.3)
          mycolors.append('tab:purple')
        
      
      part = part.split('.')[1:]
      title = ' '.join(part) # "['head', 'rotation', 'LR']" -> "head rotation LR"
      
      fig, ax = plt.subplots()
      #title
      ax.set_title(title)
      ax.pie(bins, labels = mylabels, explode = myexplode, colors=mycolors, autopct='%1.1f%%')
            
      part_img = cv2.imread('config/pic_angles/' + title + '.png')
      # check if exists
      if part_img is None:
        print(bcolors.FAIL + f"Image not found for {title}" + bcolors.ENDC)
        # print path
        print('config/pic_angles' + title + '.png')
        continue
      imagebox = OffsetImage(part_img, zoom = 0.1)#Annotation box for solar pv logo
      #Container for the imagebox referring to a specific position *xy*.
      ab = AnnotationBbox(imagebox, (-1.75,1.0), frameon = False, annotation_clip=False)
      ax.add_artist(ab)
      # plt.show()
      
      os.makedirs(os.path.join(self.output_path, 'Angles'), exist_ok=True)
      fig.savefig(os.path.join(self.output_path, 'Angles',  title + '_pie_chart.png'))
      plt.close(fig)
        
        
  def save_pie_chart_bin_score(self, aggregated_reba_score):
    parts_list = ['score.neck', 'score.trunk', 'score.shoulders', 'score.elbows', 'score.wrists', 'score.legs']
    score_dict = {}
    
    for part in parts_list:
      unique, counts = np.unique(aggregated_reba_score[part], return_counts=True)
      score_dict[part] = dict(zip(unique, counts))
      
      green_score = bin_score_per_articulation()[part]['green'] # for the neck this returns [1], and for the trunk returns [1,2]
      yellow_score = bin_score_per_articulation()[part]['yellow']
      red_score = bin_score_per_articulation()[part]['red'] # for the neck this returns [3], and for the trunk returns [4,5]
      
      green, yellow, red = 0, 0, 0
      mylabels = []
      myexplode = []
      bins = []
      mycolors = []
      
      for score in green_score:
        if score in score_dict[part]:
          green += score_dict[part][score]
          
      if green:
        bins.append(green)
        mylabels.append('Low Risk')
        myexplode.append(0)
        mycolors.append('tab:green')
        
      for score in yellow_score:
        if score in score_dict[part]:
          yellow += score_dict[part][score]
      
      if yellow:
        bins.append(yellow)
        mylabels.append('Medium Risk')
        myexplode.append(0.1)
        mycolors.append('tab:orange')
        
      for score in red_score:
        if score in score_dict[part]:
          red += score_dict[part][score]
      
      if red:
        bins.append(red)  
        mylabels.append('High Risk')
        myexplode.append(0.2)
        mycolors.append('tab:red')
      
      # plt.figure()
      fig, ax = plt.subplots()
      part = part.split('.')[1]
      ax.set_title(part)
      ax.pie(bins, labels = mylabels, explode = myexplode, colors=mycolors, autopct='%1.1f%%')
      part_img = cv2.imread('config/' + part + '.png')
      imagebox = OffsetImage(part_img, zoom = 0.3)#Annotation box for solar pv logo
      #Container for the imagebox referring to a specific position *xy*.
      ab = AnnotationBbox(imagebox, (-1.75,1.0), frameon = False, annotation_clip=False)
      ax.add_artist(ab)
      # plt.show()
      
      os.makedirs(os.path.join(self.output_path, 'Risk'), exist_ok=True)
      fig.savefig(os.path.join(self.output_path, 'Risk',  part + '_pie_chart.png'))
      plt.close(fig)
      
      
  def save_excel(self, dataframe, reba_score, aggregated_reba_score):
    dataframe.to_excel(self.writer, sheet_name='SafeMoveResults')
    reba_score.to_excel(self.writer, sheet_name='ScoreComputation')
    aggregated_reba_score.to_excel(self.writer, sheet_name='AggregatedScoreComputation')
    worksheet = self.writer.sheets['SafeMoveResults']
    
    # Write to REBA sheet
    self.save_reba_score(aggregated_reba_score)
    
    worksheet.set_column("Z:Z", 200)
    
    if True:
      for file in glob.glob(self.folder_path + 'tmp/' + '*.png'):
        row_value = from_image_name_2_excel_row_value(file)
        if row_value != -1:
          img_cell = 'Z' + str(row_value) # ColumnValue + RowValue
          worksheet.insert_image(img_cell, file, {"x_scale": 0.5, "y_scale": 0.5})
    
    self.writer.close()
    print(bcolors.OKGREEN + f"Excel saved in {self.excel_name}" + bcolors.ENDC)
    
    if True:
      try: ################# REMOVING TMP FOLDER ###################
        shutil.rmtree(self.folder_path + 'tmp/') # https://stackoverflow.com/questions/6996603/how-can-i-delete-a-file-or-folder-in-python
        print(bcolors.OKBLUE + "Deleting tmp folder" + bcolors.ENDC)
      except OSError as e:
        print(bcolors.OKBLUE + "Deleting tmp folder was unnecessary, folder not found" + bcolors.ENDC)
    

    
if __name__ == "__main__":
  parts_list = ['score.neck', 'score.trunk', 'score.shoulder', 'score.elbow', 'score.wrist', 'score.legs']
  score_dict = {}
  print(bin_score_per_articulation())
  for part in parts_list:
    green_score = bin_score_per_articulation()[part]['green']
    yellow_score = bin_score_per_articulation()[part]['yellow']
    red_score = bin_score_per_articulation()[part]['red']
    print(part)
    print(green_score, yellow_score, red_score)
    print("\n")
