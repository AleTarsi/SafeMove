import unittest
import pandas as pd
from sm_07_RiskAssessment import RiskAssessment
import numpy as np

verbose = False

def inference_test():
    gt = dict()
    results = dict()
    gt_angles = pd.read_excel('config/test.xlsx', sheet_name='Angles').fillna(0)
    print(gt_angles.index)
    gt_scores = pd.read_excel('config/test.xlsx', sheet_name='Scores').fillna(0)
    gt_reba = pd.read_excel('config/test.xlsx', sheet_name='Reba').fillna(0)
    gt_extra = pd.read_excel('config/test.xlsx', sheet_name='Extra').fillna(0)
    gt['Angles'] = gt_angles
    gt['Scores'] = gt_scores
    gt['Reba'] = gt_reba
    gt['Extra'] = gt_extra
    results['Scores'], results['Reba'] = RiskAssessment.fromDataFrame2Reba(gt['Angles'])
    return results, gt

class SafeMoveTest(unittest.TestCase):
    def test_score_excel(self):
        results, groundtruth = inference_test()
        
        for idx in range(1,groundtruth['Scores'].to_numpy().shape[0]): # Start from 1 because the first line does not contain data
            if verbose:
                print(idx, groundtruth['Scores'].iloc[idx].to_numpy(), results['Scores'].iloc[idx].to_numpy())
                print("\n\n")

            self.assertTrue(np.array_equal(groundtruth['Scores'].iloc[idx].to_numpy(), results['Scores'].iloc[idx].to_numpy())) 
            
    def test_aggregated_score_excel(self):
        results, groundtruth = inference_test()

        for idx in range(1, groundtruth['Reba'].to_numpy().shape[0]): # Start from 1 because the first line does not contain data
            if verbose:
                print(idx, groundtruth['Reba'].iloc[idx].to_numpy(), results['Reba'].iloc[idx].to_numpy()) 
                print("\n\n")

            self.assertTrue(np.array_equal(groundtruth['Reba'].iloc[idx-1].to_numpy(), results['Reba'].iloc[idx].to_numpy())) 
            
if __name__ == '__main__':
    unittest.main()
