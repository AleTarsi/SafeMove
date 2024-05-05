import pandas as pd
import numpy as np

def score_trunk_flexion_FB(val):
    new_val = val.copy()
    new_val.loc[np.where([(val>0)&(val<=20)])[1]] = 1
    new_val.loc[np.where([val>60])[1]] = 3
    new_val.loc[np.where([(val>20)&(val<=60)])[1]] = 2
    new_val.loc[np.where([val<0])[1]] = 2
    return new_val

def score_shoulder_flexion_FB(val):
    new_val = val.copy()
    new_val.loc[np.where([(val>-20)&(val<=20)])[1]] = 1
    new_val.loc[np.where([(val>20)&(val<=60)])[1]] = 2
    new_val.loc[np.where([(val>60)&(val<=90)])[1]] = 3
    new_val.loc[np.where([(val>90)])[1]] = 4
    new_val.loc[np.where([val<-20])[1]] = 2
    return new_val
    
    
def score_knee_flexion_UD(val):
    new_val = val.copy()
    new_val.loc[np.where([(val>30)&(val<=60)])[1]] = 1
    new_val.loc[np.where([(val>60)])[1]] = 2
    new_val.loc[np.where([val<=30])[1]] = 0
    return new_val
        

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
        
        aggregated_reba_score = pd.DataFrame({'t [sec]': [],
                             'score.neck': [],
                             'score.trunk': [],
                             'score.R.shoulder': [],
                             'score.L.shoulder': [],
                             'score.shoulder': [],
                             'score.R.elbow': [],
                             'score.L.elbow': [],
                             'score.elbow': [],
                             'score.R.wrist': [], 
                             'score.L.wrist': [],
                             'score.wrist': [],
                             'score.R.knee': [],
                             'score.L.knee': [],
                             'score.legs': [],
                             'score.TableA': [],
                             'score.TableB': [],
                             'score.TableC': [],
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
        
        aggregated_reba_score['t [sec]'] = pose_data['t [sec]'] 
        sum = reba_score['score.head.rotation.LR [°]'] + reba_score['score.head.flexion.DU [°]'] + reba_score['score.head.flexion.CCWCW [°]']
        aggregated_reba_score['score.neck'] = np.where(sum>3, 3, sum)
        aggregated_reba_score['score.trunk'] = reba_score['score.trunk.rotation.LR [°]'] + reba_score['score.trunk.flexion.FB [°]'] + reba_score['score.trunk.flexion.LR [°]']
        aggregated_reba_score['score.R.shoulder'] = reba_score['score.R.shoulder.flexion.FB [°]'] + reba_score['score.R.shoulder.abduction.CWCCW [°]'] 
        aggregated_reba_score['score.L.shoulder'] = reba_score['score.L.shoulder.flexion.FB [°]'] + reba_score['score.L.shoulder.abduction.CWCCW [°]'] 
        aggregated_reba_score['score.shoulder'] = aggregated_reba_score[['score.R.shoulder','score.L.shoulder']].max(axis=1) # https://stackoverflow.com/a/12169357
        aggregated_reba_score['score.R.elbow'] = reba_score['score.R.elbow.flexion.UD [°]'] 
        aggregated_reba_score['score.L.elbow'] = reba_score['score.L.elbow.flexion.UD [°]'] 
        aggregated_reba_score['score.elbow'] = aggregated_reba_score[['score.R.elbow','score.L.elbow']].max(axis=1)
        aggregated_reba_score['score.R.wrist'] = reba_score['score.R.wrist.flexion.UD [°]'] + reba_score[['score.R.elbow.rotation.PS [°]','score.R.wrist.rotation.UR [°]']].max(axis=1)
        aggregated_reba_score['score.L.wrist'] = reba_score['score.L.wrist.flexion.UD [°]'] + reba_score[['score.L.elbow.rotation.PS [°]','score.L.wrist.rotation.UR [°]']].max(axis=1)
        aggregated_reba_score['score.wrist'] = aggregated_reba_score[['score.R.wrist','score.L.wrist']].max(axis=1)
        aggregated_reba_score['score.R.knee'] = reba_score['score.R.knee.flexion.UD [°]'] + reba_score['score.contact points [#]']
        aggregated_reba_score['score.L.knee'] = reba_score['score.L.knee.flexion.UD [°]'] + reba_score['score.contact points [#]']
        aggregated_reba_score['score.legs'] = aggregated_reba_score[['score.R.knee','score.L.knee']].max(axis=1)
        
        Table_A = np.zeros((3,5,4)) # the first position is for Neck, second position is for trunk, the third position is for legs
        Table_A[0,0,0] = 1
        Table_A[0,0,1] = 2
        Table_A[0,0,2] = 3
        Table_A[0,0,3] = 4
        Table_A[0,1,0] = 2
        Table_A[0,1,1] = 3
        Table_A[0,1,2] = 4
        Table_A[0,1,3] = 5
        Table_A[0,2,0] = 2
        Table_A[0,2,1] = 4
        Table_A[0,2,2] = 5
        Table_A[0,2,3] = 6
        Table_A[0,3,0] = 3
        Table_A[0,3,1] = 5
        Table_A[0,3,2] = 6
        Table_A[0,3,3] = 7
        Table_A[0,4,0] = 4
        Table_A[0,4,1] = 6
        Table_A[0,4,2] = 7
        Table_A[0,4,3] = 8
        Table_A[1,0,0] = 1
        Table_A[1,0,1] = 2
        Table_A[1,0,2] = 3
        Table_A[1,0,3] = 4
        Table_A[1,1,0] = 3
        Table_A[1,1,1] = 4
        Table_A[1,1,2] = 5
        Table_A[1,1,3] = 6
        Table_A[1,2,0] = 4
        Table_A[1,2,1] = 5
        Table_A[1,2,2] = 6
        Table_A[1,2,3] = 7
        Table_A[1,3,0] = 5
        Table_A[1,3,1] = 6
        Table_A[1,3,2] = 7
        Table_A[1,3,3] = 8
        Table_A[1,4,0] = 6
        Table_A[1,4,1] = 7
        Table_A[1,4,2] = 8
        Table_A[1,4,3] = 9
        Table_A[2,0,0] = 3
        Table_A[2,0,1] = 3
        Table_A[2,0,2] = 5 
        Table_A[2,0,3] = 6
        Table_A[2,1,0] = 4
        Table_A[2,1,1] = 5
        Table_A[2,1,2] = 6
        Table_A[2,1,3] = 7
        Table_A[2,2,0] = 5
        Table_A[2,2,1] = 6
        Table_A[2,2,2] = 7
        Table_A[2,2,3] = 8
        Table_A[2,3,0] = 6
        Table_A[2,3,1] = 7
        Table_A[2,3,2] = 8
        Table_A[2,3,3] = 9
        Table_A[2,4,0] = 7
        Table_A[2,4,1] = 8
        Table_A[2,4,2] = 9
        Table_A[2,4,3] = 9
        
        Table_B = np.zeros((2,6,3)) # the first position is for Lower arm, second position is for Upper Arm, the third position is for wrist
        Table_B[0,0,0] = 1 
        Table_B[0,0,1] = 2
        Table_B[0,0,2] = 2
        Table_B[0,1,0] = 1
        Table_B[0,1,1] = 2
        Table_B[0,1,2] = 3
        Table_B[0,2,0] = 3
        Table_B[0,2,1] = 4
        Table_B[0,2,2] = 5
        Table_B[0,3,0] = 4
        Table_B[0,3,1] = 5
        Table_B[0,3,2] = 5
        Table_B[0,4,0] = 6
        Table_B[0,4,1] = 7
        Table_B[0,4,2] = 8
        Table_B[0,5,0] = 7
        Table_B[0,5,1] = 8
        Table_B[0,5,2] = 8
        Table_B[1,0,0] = 1 
        Table_B[1,0,1] = 2
        Table_B[1,0,2] = 3
        Table_B[1,1,0] = 2
        Table_B[1,1,1] = 3
        Table_B[1,1,2] = 4
        Table_B[1,2,0] = 4
        Table_B[1,2,1] = 5
        Table_B[1,2,2] = 5
        Table_B[1,3,0] = 5
        Table_B[1,3,1] = 6
        Table_B[1,3,2] = 7
        Table_B[1,4,0] = 7
        Table_B[1,4,1] = 8
        Table_B[1,4,2] = 8
        Table_B[1,5,0] = 8
        Table_B[1,5,1] = 9
        Table_B[1,5,2] = 9
        
        Table_C = np.zeros((12,12)) # the first position is for table A score, second position is for table B score
        Table_C[0,0] = 1
        Table_C[0,1] = 1
        Table_C[0,2] = 1
        Table_C[0,3] = 2
        Table_C[0,4] = 3
        Table_C[0,5] = 3
        Table_C[0,6] = 4
        Table_C[0,7] = 5
        Table_C[0,8] = 6
        Table_C[0,9] = 7
        Table_C[0,10] = 7
        Table_C[0,11] = 7
        Table_C[1,0] = 1
        Table_C[1,1] = 2
        Table_C[1,2] = 2
        Table_C[1,3] = 3
        Table_C[1,4] = 4
        Table_C[1,5] = 4
        Table_C[1,6] = 5
        Table_C[1,7] = 6
        Table_C[1,8] = 6
        Table_C[1,9] = 7
        Table_C[1,10] = 7
        Table_C[1,11] = 8
        Table_C[2,0] = 2
        Table_C[2,1] = 3
        Table_C[2,2] = 3
        Table_C[2,3] = 3
        Table_C[2,4] = 4
        Table_C[2,5] = 5
        Table_C[2,6] = 6
        Table_C[2,7] = 7
        Table_C[2,8] = 7
        Table_C[2,9] = 8
        Table_C[2,10] = 8
        Table_C[2,11] = 8
        Table_C[3,0] = 3
        Table_C[3,1] = 4
        Table_C[3,2] = 4
        Table_C[3,3] = 4
        Table_C[3,4] = 5
        Table_C[3,5] = 6
        Table_C[3,6] = 7
        Table_C[3,7] = 8
        Table_C[3,8] = 8
        Table_C[3,9] = 9
        Table_C[3,10] = 9
        Table_C[3,11] = 9
        Table_C[4,0] = 4
        Table_C[4,1] = 4
        Table_C[4,2] = 4
        Table_C[4,3] = 5
        Table_C[4,4] = 6
        Table_C[4,5] = 7
        Table_C[4,6] = 8
        Table_C[4,7] = 8
        Table_C[4,8] = 9
        Table_C[4,9] = 9
        Table_C[4,10] = 9
        Table_C[4,11] = 9
        Table_C[5,0] = 6
        Table_C[5,1] = 6
        Table_C[5,2] = 6
        Table_C[5,3] = 7
        Table_C[5,4] = 8
        Table_C[5,5] = 8
        Table_C[5,6] = 9
        Table_C[5,7] = 9
        Table_C[5,8] = 10
        Table_C[5,9] = 10
        Table_C[5,10] = 10
        Table_C[5,11] = 10
        Table_C[6,0] = 7
        Table_C[6,1] = 7
        Table_C[6,2] = 7
        Table_C[6,3] = 8
        Table_C[6,4] = 9
        Table_C[6,5] = 9
        Table_C[6,6] = 9
        Table_C[6,7] = 10
        Table_C[6,8] = 10
        Table_C[6,9] = 11
        Table_C[6,10] = 11
        Table_C[6,11] = 11
        Table_C[7,0] = 8
        Table_C[7,1] = 8
        Table_C[7,2] = 8
        Table_C[7,3] = 9
        Table_C[7,4] = 10
        Table_C[7,5] = 10
        Table_C[7,6] = 10
        Table_C[7,7] = 10
        Table_C[7,8] = 10
        Table_C[7,9] = 11
        Table_C[7,10] = 11
        Table_C[7,11] = 11
        Table_C[8,0] = 9
        Table_C[8,1] = 9
        Table_C[8,2] = 9
        Table_C[8,3] = 10
        Table_C[8,4] = 10
        Table_C[8,5] = 10
        Table_C[8,6] = 11
        Table_C[8,7] = 11
        Table_C[8,8] = 11
        Table_C[8,9] = 12
        Table_C[8,10] = 12
        Table_C[8,11] = 12
        Table_C[9,0] = 10
        Table_C[9,1] = 10
        Table_C[9,2] = 10
        Table_C[9,3] = 11
        Table_C[9,4] = 11
        Table_C[9,5] = 11
        Table_C[9,6] = 11 
        Table_C[9,7] = 12
        Table_C[9,8] = 12
        Table_C[9,9] = 12
        Table_C[9,10] = 12
        Table_C[9,11] = 12
        Table_C[10,0] = 11
        Table_C[10,1] = 11 
        Table_C[10,2] = 11
        Table_C[10,3] = 11
        Table_C[10,4] = 11
        Table_C[10,5] = 12
        Table_C[10,6] = 12
        Table_C[10,7] = 12
        Table_C[10,8] = 12
        Table_C[10,9] = 12
        Table_C[10,10] = 12
        Table_C[10,11] = 12
        Table_C[11,0] = 12
        Table_C[11,1] = 12
        Table_C[11,2] = 12
        Table_C[11,3] = 12
        Table_C[11,4] = 12
        Table_C[11,5] = 12
        Table_C[11,6] = 12
        Table_C[11,7] = 12
        Table_C[11,8] = 12
        Table_C[11,9] = 12
        Table_C[11,10] = 12
        Table_C[11,11] = 12
        
        # pd.melt
        # pd.factorize
        
        pd.options.mode.chained_assignment = None  # default='warn'
        
        for idx in range(len(aggregated_reba_score['score.neck'])):
            aggregated_reba_score['score.TableA'].loc[idx] = Table_A()[str(int(aggregated_reba_score['score.neck'].loc[idx]-1))][str(int(aggregated_reba_score['score.trunk'].loc[idx]-1))][str(int(aggregated_reba_score['score.legs'].loc[idx]-1))]
            # aggregated_reba_score['score.TableB'].loc[idx] = Table_B[int(aggregated_reba_score['score.neck'].loc[idx]-1),int(aggregated_reba_score['score.trunk'].loc[idx]-1),int(aggregated_reba_score['score.legs'].loc[idx]-1)]
            
        
        return reba_score, aggregated_reba_score

        