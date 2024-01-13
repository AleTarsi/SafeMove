''' 
This file contains the API to export the data to an excel file
'''
import pandas as pd


class Result:
  
  def __init__(self, source_video):
    self.source_video = source_video
  '''
  We followed the structure and the order present in the excel
  For what regards the Data Frame the first letter indicates the positive direction and the second the negative direction
  '''
  error_list = pd.DataFrame({'t [sec]': [],
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

    
    
    