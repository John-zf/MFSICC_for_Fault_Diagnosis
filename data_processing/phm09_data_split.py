# -*- coding: utf-8 -*-
"""
Created on 2019/11/20 19:29

@author: John_Fengz
"""
import numpy as np
import pandas as pd


WIN_SIZE = 1024
file_name = 'helical 1_40hz_Low_2'
file_folder = 'F:/MyResearch/PHM/Experiments/2 GroupL21/Datasets/PHM09_Gearbox/Raw data' \
              '/Low/Normal/'
file_path = file_folder + file_name + '.txt'

with open(file_path, 'r') as f:
    data = f.readlines()
data = [x.strip().split(' ') for x in data]
data_clean = []
for ins in data:
    ins_re = [i for i in ins if i != '']
    ins_re = [float(i.strip()) for i in ins_re]
    data_clean.append(ins_re)

col2 = np.array(data_clean)[:, 1]
xxx = col2[:-(len(col2) % WIN_SIZE)].reshape((int(len(col2)/WIN_SIZE), WIN_SIZE))
df = pd.DataFrame(xxx)
df.to_csv(file_folder+file_name+'.csv', index=False, header=False, encoding='utf-8')
