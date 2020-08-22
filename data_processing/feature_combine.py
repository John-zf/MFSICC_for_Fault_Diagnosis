# -*- coding: utf-8 -*-
"""
Created on 2019/11/25 9:33

@author: John_Fengz
"""
import numpy as np
import pandas as pd

folder = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Features\\'
file1 = 'Multi-Domain\\CWRU_HP3_Domain.csv'
file2 = 'SAE\\CWRU_HP3_SAE.csv'

file3 = 'Features_All\\CWRU_HP3_COMB.csv'

data1 = pd.read_csv(folder+file1)
feature1 = np.array(data1.iloc[:, :-1])
data2 = np.array(pd.read_csv(folder+file2))

data3 = np.hstack((feature1, data2))
df = pd.DataFrame(data3)
df.to_csv(folder+file3, index=None)
