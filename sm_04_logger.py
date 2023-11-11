''' 
This file contains the API to export the data to an excel file
'''
import pandas as pd


class Result:
  
  def __init__(self, source_video):
    self.source_video = source_video
    
  error_list = pd.DataFrame({'t': [],
                             'head_x': [],
                             'head_y': [],
                             'trunk_x': [],
                             'trunk_y': [],
                            })

    
    
    