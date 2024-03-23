import pandas as pd
import numpy as np

def score_trunk_flexion_FB(val):
    new_val = val
    new_val.loc[np.where([(val>0)&(val<=20)])[1]] = 1
    new_val.loc[np.where([val>60])[1]] = 3
    new_val.loc[np.where([(val>20)&(val<=60)])[1]] = 2
    new_val.loc[np.where([val<0])[1]] = 2
    return val

def score_shoulder_flexion_FB(val):
    new_val = val
    new_val.loc[np.where([(val>-20)&(val<=20)])[1]] = 1
    new_val.loc[np.where([(val>20)&(val<=60)])[1]] = 2
    new_val.loc[np.where([(val>60)&(val<=90)])[1]] = 3
    new_val.loc[np.where([(val>90)])[1]] = 4
    new_val.loc[np.where([val<-20])[1]] = 2
    return val
    
    
def score_knee_flexion_UD(val):
    new_val = val
    new_val.loc[np.where([(val>30)&(val<=60)])[1]] = 1
    new_val.loc[np.where([(val>60)])[1]] = 2
    new_val.loc[np.where([val<=30])[1]] = 0
    return val
        

class RiskAssessment:
    def fromDataFrame2Reba(pose_data):
        
        reba_score = pd.DataFrame({'t [sec]': [],
                             'score.head.rotation.LR [°]': [],
                             'score.head.flexion.DU [°]': [],
                             'score.head.flexion.CCWCW [°]': [],
                             'score.trunk.rotation.LR [°]': [],
                             'score.trunk.flexion.FB [°]': [],
                             'score.trunk.flexion.LR [°]': [],
                             'score.R.shoulder.flexion.FB [°]': [],
                             'score.R.shoulder.abduction.CWCCW [°]': [],
                             'score.L.shoulder.flexion.FB [°]': [],
                             'score.L.shoulder.abduction.CWCCW [°]': [],
                             'score.R.elbow.flexion.UD [°]': [],
                             'score.R.elbow.rotation.PS [°]': [],
                             'score.L.elbow.flexion.UD [°]': [],
                             'score.L.elbow.rotation.PS [°]': [],
                             'score.R.wrist.flexion.UD [°]': [],
                             'score.R.wrist.rotation.UR [°]': [], 
                             'score.L.wrist.flexion.UD [°]': [],
                             'score.L.wrist.rotation.UR [°]': [],
                             'score.R.knee.flexion.UD [°]': [],
                             'score.L.knee.flexion.UD [°]': [],
                             'score.contact points [#]': [],
                             'score.R.ankle.flexion.UD [°]': [],
                             'score.L.ankle.flexion.UD [°]': [],
                            })
        
        # Have a look at the following link to see why we avoided using "in range" function: https://stackoverflow.com/questions/36921951/truth-value-of-a-series-is-ambiguous-use-a-empty-a-bool-a-item-a-any-o
        reba_score['t [sec]'] = pose_data['t [sec]']
        reba_score['score.head.rotation.LR [°]'] = np.where(abs(pose_data['head.rotation.LR [°]'])<45, 0, 1)
        reba_score['score.head.flexion.DU [°]'] =  np.where((pose_data['head.flexion.DU [°]']>0) & (pose_data['head.flexion.DU [°]']<20), 1,2)
        reba_score['score.head.flexion.CCWCW [°]'] = np.where(abs(pose_data['head.flexion.CCWCW [°]'])<10, 0, 1)
        reba_score['score.trunk.rotation.LR [°]'] = np.where(abs(pose_data['trunk.rotation.LR [°]'])<10, 0, 1)
        reba_score['score.trunk.flexion.FB [°]'] = score_trunk_flexion_FB(pose_data['trunk.flexion.FB [°]'])
        reba_score['score.trunk.flexion.LR [°]'] = np.where(abs(pose_data['trunk.flexion.LR [°]'])<10, 0, 1)
        reba_score['score.R.shoulder.flexion.FB [°]'] = score_shoulder_flexion_FB(pose_data['R.shoulder.flexion.FB [°]'])
        reba_score['score.R.shoulder.abduction.CWCCW [°]'] = np.where(abs(pose_data['R.shoulder.abduction.CWCCW [°]'])<60, 0, 1)
        reba_score['score.L.shoulder.flexion.FB [°]'] = score_shoulder_flexion_FB(pose_data['L.shoulder.flexion.FB [°]'])
        reba_score['score.L.shoulder.abduction.CWCCW [°]'] = np.where(abs(pose_data['L.shoulder.abduction.CWCCW [°]'])<60, 0, 1)
        reba_score['score.R.elbow.flexion.UD [°]'] = np.where((pose_data['R.elbow.flexion.UD [°]']>30)&(pose_data['R.elbow.flexion.UD [°]']<170), 1,2)
        reba_score['score.R.elbow.rotation.PS [°]'] = np.where(abs(pose_data['R.elbow.rotation.PS [°]'])<60, 0, 1)
        reba_score['score.L.elbow.flexion.UD [°]'] = np.where((pose_data['L.elbow.flexion.UD [°]']>30)&(pose_data['L.elbow.flexion.UD [°]']<170), 1,2)
        reba_score['score.L.elbow.rotation.PS [°]'] = np.where(abs(pose_data['L.elbow.rotation.PS [°]'])<60, 0, 1)
        reba_score['score.R.wrist.flexion.UD [°]'] = np.where((pose_data['R.wrist.flexion.UD [°]']>-15) & (pose_data['R.wrist.flexion.UD [°]']<15), 0, 1)
        reba_score['score.R.wrist.rotation.UR [°]'] = np.where((pose_data['R.wrist.rotation.UR [°]']>-24) & (pose_data['R.wrist.rotation.UR [°]']<15), 0, 1)
        reba_score['score.L.wrist.flexion.UD [°]'] = np.where((pose_data['L.wrist.flexion.UD [°]']>-15) & (pose_data['L.wrist.flexion.UD [°]']<15), 0, 1)
        reba_score['score.L.wrist.rotation.UR [°]'] = np.where((pose_data['L.wrist.rotation.UR [°]']>-24) & (pose_data['L.wrist.rotation.UR [°]']<15), 0, 1)
        reba_score['score.R.knee.flexion.UD [°]'] = score_knee_flexion_UD(pose_data['R.knee.flexion.UD [°]'])
        reba_score['score.L.knee.flexion.UD [°]'] = score_knee_flexion_UD(pose_data['L.knee.flexion.UD [°]'])
        reba_score['score.contact points [#]'] = pose_data['contact points [#]']
        
        reba_score['score.R.ankle.flexion.UD [°]'] = pose_data['R.ankle.flexion.UD [°]']
        reba_score['score.L.ankle.flexion.UD [°]'] = pose_data['L.ankle.flexion.UD [°]']
        
        return reba_score

        