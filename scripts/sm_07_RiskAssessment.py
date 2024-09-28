# MIT License
#
# Copyright (c) [YEAR] [Your Name or Your Organization]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pandas as pd
import numpy as np
from sm_10_reba_tables import Table_A, Table_B, Table_C

def score_trunk_flexion_FB(val):
    new_val = val.copy()
    new_val.loc[np.where([(val>0)&(val<=20)])[1]] = 1
    new_val.loc[np.where([val>60])[1]] = 3
    new_val.loc[np.where([(val>20)&(val<=60)])[1]] = 2
    new_val.loc[np.where([val<0])[1]] = 2
    return new_val.astype(int)

def score_shoulder_flexion_FB(val):
    new_val = val.copy()
    new_val.loc[np.where([(val>-20)&(val<=20)])[1]] = 1
    new_val.loc[np.where([(val>20)&(val<=60)])[1]] = 2
    new_val.loc[np.where([(val>60)&(val<=90)])[1]] = 3
    new_val.loc[np.where([(val>90)])[1]] = 4
    new_val.loc[np.where([val<-20])[1]] = 2
    return new_val.astype(int)
    
    
def score_knee_flexion_UD(val):
    new_val = val.copy()
    new_val.loc[np.where([(val>30)&(val<=60)])[1]] = 1
    new_val.loc[np.where([(val>60)])[1]] = 2
    new_val.loc[np.where([val<=30])[1]] = 0
    return new_val.astype(int)
        

class RiskAssessment:
    def fromDataFrame2Reba(pose_data, Force, Coupling, Activity):
        
        reba_score = pd.DataFrame({'t [sec]': [],
                             'score.head.rotation.LR': [],
                             'score.head.flexion.DU': [],
                             'score.head.flexion.CCWCW': [],
                             'score.trunk.rotation.LR': [],
                             'score.trunk.flexion.FB': [],
                             'score.trunk.flexion.LR': [],
                             'score.R.shoulder.flexion.FB': [],
                             'score.R.shoulder.abduction.CWCCW': [],
                             'score.L.shoulder.flexion.FB': [],
                             'score.L.shoulder.abduction.CWCCW': [],
                             'score.R.elbow.flexion.UD': [],
                             'score.R.elbow.rotation.PS': [],
                             'score.L.elbow.flexion.UD': [],
                             'score.L.elbow.rotation.PS': [],
                             'score.R.wrist.flexion.UD': [],
                             'score.R.wrist.rotation.UR': [], 
                             'score.L.wrist.flexion.UD': [],
                             'score.L.wrist.rotation.UR': [],
                             'score.R.knee.flexion.UD': [],
                             'score.L.knee.flexion.UD': [],
                             'score.contact points [#]': [],
                            })
        
        aggregated_reba_score = pd.DataFrame({'t [sec]': [],
                             'score.neck': [],
                             'score.trunk': [],
                             'score.R.shoulder': [],
                             'score.L.shoulder': [],
                             'score.shoulders': [],
                             'score.R.elbow': [],
                             'score.L.elbow': [],
                             'score.elbows': [],
                             'score.R.wrist': [], 
                             'score.L.wrist': [],
                             'score.wrists': [],
                             'score.R.leg': [],
                             'score.L.leg': [],
                             'score.legs': [],
                             'score.TableA': [],
                             'score.TableB': [],
                             'score.TableA.mean': [],
                             'score.TableB.mean': [],
                             'score.Force': [],
                             'score.Coupling': [],
                             'score.TableA.Tot': [],
                             'score.TableB.Tot': [],
                             'score.TableC': [],
                             'score.Activity': [],
                             'score.REBA': [],
                             'score.trunk.Tot': [],
                             'score.leg.Tot': [],
                             'score.elbow.Tot': [],
                             'score.wrist.Tot': [],
                             'score.shoulder.Tot': [],
                             'score.neck.Tot': [],
                            })
        
        # Have a look at the following link to see why we avoided using "in range" function: https://stackoverflow.com/questions/36921951/truth-value-of-a-series-is-ambiguous-use-a-empty-a-bool-a-item-a-any-o
        reba_score['t [sec]'] = pose_data['t [sec]'] 
        reba_score['score.head.rotation.LR'] = np.where(abs(pose_data['head.rotation.LR'])<45, 0, 1)
        reba_score['score.head.flexion.DU'] =  np.where((pose_data['head.flexion.DU']>0) & (pose_data['head.flexion.DU']<=20), 1,2)
        reba_score['score.head.flexion.CCWCW'] = np.where(abs(pose_data['head.flexion.CCWCW'])<=10, 0, 1)
        reba_score['score.trunk.rotation.LR'] = np.where(abs(pose_data['trunk.rotation.LR'])<=10, 0, 1)
        reba_score['score.trunk.flexion.FB'] = score_trunk_flexion_FB(pose_data['trunk.flexion.FB'])
        reba_score['score.trunk.flexion.LR'] = np.where(abs(pose_data['trunk.flexion.LR'])<=10, 0, 1)
        reba_score['score.R.shoulder.flexion.FB'] = score_shoulder_flexion_FB(pose_data['R.shoulder.flexion.FB'])
        reba_score['score.R.shoulder.abduction.CWCCW'] = np.where(abs(pose_data['R.shoulder.abduction.CWCCW'])<=60, 0, 1)
        reba_score['score.L.shoulder.flexion.FB'] = score_shoulder_flexion_FB(pose_data['L.shoulder.flexion.FB'])
        reba_score['score.L.shoulder.abduction.CWCCW'] = np.where(abs(pose_data['L.shoulder.abduction.CWCCW'])<=60, 0, 1)
        reba_score['score.R.elbow.flexion.UD'] = np.where((pose_data['R.elbow.flexion.UD']>=30)&(pose_data['R.elbow.flexion.UD']<=170), 1,2)
        reba_score['score.R.elbow.rotation.PS'] = np.where(abs(pose_data['R.elbow.rotation.PS'])<=60, 0, 1)
        reba_score['score.L.elbow.flexion.UD'] = np.where((pose_data['L.elbow.flexion.UD']>=30)&(pose_data['L.elbow.flexion.UD']<=170), 1,2)
        reba_score['score.L.elbow.rotation.PS'] = np.where(abs(pose_data['L.elbow.rotation.PS'])<=60, 0, 1)
        reba_score['score.R.wrist.flexion.UD'] = np.where((pose_data['R.wrist.flexion.UD']>=-15) & (pose_data['R.wrist.flexion.UD']<=15), 1, 2)
        reba_score['score.R.wrist.rotation.UR'] = np.where((pose_data['R.wrist.rotation.UR']>=-15) & (pose_data['R.wrist.rotation.UR']<=24), 0, 1)
        reba_score['score.L.wrist.flexion.UD'] = np.where((pose_data['L.wrist.flexion.UD']>=-15) & (pose_data['L.wrist.flexion.UD']<=15), 1, 2)
        reba_score['score.L.wrist.rotation.UR'] = np.where((pose_data['L.wrist.rotation.UR']>=-15) & (pose_data['L.wrist.rotation.UR']<=24), 0, 1)
        reba_score['score.R.knee.flexion.UD'] = score_knee_flexion_UD(pose_data['R.knee.flexion.UD'])
        reba_score['score.L.knee.flexion.UD'] = score_knee_flexion_UD(pose_data['L.knee.flexion.UD'])
        reba_score['score.contact points [#]'] = pose_data['contact points [#]']
        
        aggregated_reba_score['t [sec]'] = pose_data['t [sec]'] 
        head_sum = reba_score['score.head.rotation.LR'] + reba_score['score.head.flexion.DU'] + reba_score['score.head.flexion.CCWCW']
        aggregated_reba_score['score.neck'] = np.where(head_sum>3, 3, head_sum)
        aggregated_reba_score['score.trunk'] = reba_score['score.trunk.rotation.LR'] + reba_score['score.trunk.flexion.FB'] + reba_score['score.trunk.flexion.LR']
        aggregated_reba_score['score.R.shoulder'] = reba_score['score.R.shoulder.flexion.FB'] + reba_score['score.R.shoulder.abduction.CWCCW'] 
        aggregated_reba_score['score.L.shoulder'] = reba_score['score.L.shoulder.flexion.FB'] + reba_score['score.L.shoulder.abduction.CWCCW'] 
        aggregated_reba_score['score.shoulders'] = aggregated_reba_score[['score.R.shoulder','score.L.shoulder']].max(axis=1) # https://stackoverflow.com/a/12169357
        aggregated_reba_score['score.R.elbow'] = reba_score['score.R.elbow.flexion.UD'] 
        aggregated_reba_score['score.L.elbow'] = reba_score['score.L.elbow.flexion.UD'] 
        aggregated_reba_score['score.elbows'] = aggregated_reba_score[['score.R.elbow','score.L.elbow']].max(axis=1)
        aggregated_reba_score['score.R.wrist'] = reba_score['score.R.wrist.flexion.UD'] + reba_score[['score.R.elbow.rotation.PS','score.R.wrist.rotation.UR']].max(axis=1)
        aggregated_reba_score['score.L.wrist'] = reba_score['score.L.wrist.flexion.UD'] + reba_score[['score.L.elbow.rotation.PS','score.L.wrist.rotation.UR']].max(axis=1)
        aggregated_reba_score['score.wrists'] = aggregated_reba_score[['score.R.wrist','score.L.wrist']].max(axis=1)
        aggregated_reba_score['score.R.leg'] = reba_score['score.R.knee.flexion.UD'] + reba_score['score.contact points [#]']
        aggregated_reba_score['score.L.leg'] = reba_score['score.L.knee.flexion.UD'] + reba_score['score.contact points [#]']
        aggregated_reba_score['score.legs'] = aggregated_reba_score[['score.R.leg','score.L.leg']].max(axis=1)
        
        pd.options.mode.chained_assignment = None  # default='warn'
        
        for idx in range(len(aggregated_reba_score['score.neck'])):
            try:
                # We subtract 1, because we started to count from 0 in the tables
                aggregated_reba_score['score.TableA'].loc[idx] = Table_A()[str(int(aggregated_reba_score['score.neck'].loc[idx]-1))+str(int(aggregated_reba_score['score.trunk'].loc[idx]-1))+str(int(aggregated_reba_score['score.legs'].loc[idx]-1))]
                aggregated_reba_score['score.TableB'].loc[idx] = Table_B()[str(int(aggregated_reba_score['score.elbows'].loc[idx]-1))+str(int(aggregated_reba_score['score.shoulders'].loc[idx]-1))+str(int(aggregated_reba_score['score.wrists'].loc[idx]-1))]
            except:
                print("Error computing the Table Value")
                print("This error is thrown during the unittest due to the first line of the DF which is empty.")
        
        # have a look at the following link to see why we used loc like this: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        aggregated_reba_score.loc[1, 'score.TableA.mean'] = np.nanmean(aggregated_reba_score['score.TableA'].to_numpy())
        aggregated_reba_score.loc[1, 'score.TableB.mean'] = np.nanmean(aggregated_reba_score['score.TableB'].to_numpy())
        aggregated_reba_score.loc[1, 'score.Force'] = Force
        aggregated_reba_score.loc[1, 'score.Coupling'] = Coupling
        aggregated_reba_score.loc[1, 'score.TableA.Tot'] = round(aggregated_reba_score.loc[1, 'score.TableA.mean'] + Force)
        aggregated_reba_score.loc[1, 'score.TableB.Tot'] = round(aggregated_reba_score.loc[1, 'score.TableB.mean'] + Coupling)
        aggregated_reba_score.loc[1, 'score.TableC'] = Table_C()[str(round(aggregated_reba_score.loc[1, 'score.TableA.Tot'])-1)+str(round(aggregated_reba_score.loc[1, 'score.TableB.Tot'])-1)] 
        aggregated_reba_score.loc[1, 'score.Activity'] = Activity
        aggregated_reba_score.loc[1, 'score.REBA'] = aggregated_reba_score.loc[1, 'score.TableC'] + Activity
        aggregated_reba_score.loc[1, 'score.trunk.Tot'] = np.nanmean(aggregated_reba_score['score.trunk'].to_numpy())
        aggregated_reba_score.loc[1, 'score.leg.Tot'] = np.nanmean(aggregated_reba_score['score.legs'].to_numpy())
        aggregated_reba_score.loc[1, 'score.elbow.Tot'] = np.nanmean(aggregated_reba_score['score.elbows'].to_numpy())
        aggregated_reba_score.loc[1, 'score.wrist.Tot'] = np.nanmean(aggregated_reba_score['score.wrists'].to_numpy())
        aggregated_reba_score.loc[1, 'score.shoulder.Tot'] = np.nanmean(aggregated_reba_score['score.shoulders'].to_numpy())   
        aggregated_reba_score.loc[1, 'score.neck.Tot'] = np.nanmean(aggregated_reba_score['score.neck'].to_numpy())
        
        return reba_score, aggregated_reba_score
        