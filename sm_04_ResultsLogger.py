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

class ResultsLogger:
  
  def __init__(self, folder_path, source_video):
    current_time = datetime.datetime.now()
    self.folder_path = folder_path
    excel_path = folder_path + '/excel/' + source_video
    self.excel_name = excel_path + '/' + str(current_time.year) + '_' + str(current_time.month) + '_' + str(current_time.day) + '__' + str(current_time.hour) + '_' + str(current_time.minute) + '_' + str(current_time.second) + '.xlsx' 
    
    # Create folder if it does not exists
    if not os.path.exists(excel_path):
      os.makedirs(excel_path)
    
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
    
    
  def save_excel(self, dataframe, reba_score):
    dataframe.to_excel(self.writer, sheet_name='SafeMoveResults')
    reba_score.to_excel(self.writer, sheet_name='RangeComputation')
    worksheet = self.writer.sheets['SafeMoveResults']
    
    worksheet.set_column("Z:Z", 200)
    
    if  True:
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
    

    
    
    