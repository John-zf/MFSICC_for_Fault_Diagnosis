# -*- coding: utf-8 -*-
"""
Created on 2019/11/21 9:42

@author: John_Fengz
"""
import numpy as np
import pandas as pd
import os
import random


file_num_each = 200
labels = ['Normal', 'MissingTooth', 'Clipped']
folder_name = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Datasets\\PHM09_Gearbox\\Spilt_data\\Low\\'
write_file = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Features\\PHM09_Gearbox\\Low\\PHM09_Low_Raw.csv'

data = []
for k, label in enumerate(labels):
    file_folder = folder_name + '\\' + label
    files = os.listdir(file_folder)

    for file in files:
        file_path = file_folder + '\\' + file
        df = pd.read_csv(file_path, header=None)
        content = np.array(df)
        idx = np.arange(len(content))
        random.shuffle(idx, random=random.seed(22580))
        content_used = content[idx[:file_num_each], :]
        label_name = np.repeat(k, file_num_each)
        content_conc = np.hstack((content_used, label_name[:, np.newaxis]))
        data.append(content_conc)

    print(label)

df = pd.DataFrame(np.vstack(data))
df.to_csv(write_file, index=False)
