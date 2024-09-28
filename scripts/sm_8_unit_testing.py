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
    results['Scores'], results['Reba'] = RiskAssessment.fromDataFrame2Reba(gt['Angles'], Force=3, Coupling=2, Activity=1) # We hardcoded these values also in the excel
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
                    
            if idx == 1:
                self.assertTrue(np.array_equal(groundtruth['Reba'].iloc[idx].to_numpy(), results['Reba'].iloc[idx].to_numpy())) 
            else:
                self.assertTrue(np.array_equal(groundtruth['Reba'].iloc[idx].to_numpy()[:17], results['Reba'].iloc[idx].to_numpy()[:17])) # The empty cells in one format are considered as 0, and in the other format are considered as NaN. Thus, we prune them
                
if __name__ == '__main__':
    unittest.main()
