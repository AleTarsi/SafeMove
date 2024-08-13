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

class ResultsLogger:
  
  def __init__(self, folder_path, output_path):
    self.folder_path = folder_path
    
    # Create folder if it does not exists
    if not os.path.exists(output_path):
      os.makedirs(output_path)
      
    self.reba_image_path = os.path.join(output_path, 'reba.png')
      
    self.excel_name = output_path + '/SafeMoveResults.xlsx'
    
    self.writer = pd.ExcelWriter(self.excel_name , engine='xlsxwriter')
  
  '''
  We followed the structure and the order present in the excel
  For what regards the Data Frame the first letter indicates the positive direction and the second the negative direction
  '''
  pose_data = pd.DataFrame({'t [sec]': [],
                             'head.rotation.LR [°]': [],
                             'head.flexion.DU [°]': [],
                             'head.flexion.CCWCW [°]': [],
                             'trunk.rotation.LR [°]': [],
                             'trunk.flexion.FB [°]': [],
                             'trunk.flexion.LR [°]': [],
                             'R.shoulder.flexion.FB [°]': [],
                             'R.shoulder.abduction.CWCCW [°]': [],
                             'L.shoulder.flexion.FB [°]': [],
                             'L.shoulder.abduction.CWCCW [°]': [],
                             'R.elbow.flexion.UD [°]': [],
                             'R.elbow.rotation.PS [°]': [],
                             'L.elbow.flexion.UD [°]': [],
                             'L.elbow.rotation.PS [°]': [],
                             'R.wrist.flexion.UD [°]': [],
                             'R.wrist.rotation.UR [°]': [], 
                             'L.wrist.flexion.UD [°]': [],
                             'L.wrist.rotation.UR [°]': [],
                             'R.knee.flexion.UD [°]': [],
                             'L.knee.flexion.UD [°]': [],
                             'contact points [#]': [],
                             'R.ankle.flexion.UD [°]': [],
                             'L.ankle.flexion.UD [°]': [],
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
      if not os.path.exists(self.folder_path + 'tmp/'):
        os.makedirs( self.folder_path + 'tmp/')

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
    
  def save_pie_chart(self, aggregated_reba_score):
    parts_list = ['score.neck'] #, 'score.trunk', 'score.R.shoulder', 'score.L.shoulder']
    score_dict = {}
    
    for part in parts_list:
      unique, counts = np.unique(aggregated_reba_score[part], return_counts=True)
      score_dict[part] = dict(zip(unique, counts))
      
      green = score_dict[part][0]
      yellow = score_dict[part][1]
      total = green + yellow
      
      # Draw pie chart
      
      
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
    

    
    
    