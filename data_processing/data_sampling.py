# -*- coding: utf-8 -*-
"""
Created on 2019/11/29 12:46

@author: John_Fengz
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

write_path = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Datasets\\Datasets_Sample200\\CWRU_HP1_Raw.csv'
file_path = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Datasets\\Datasets_All\\CWRU_HP1_Raw.csv'
data = pd.read_csv(file_path)

features = np.array(data.iloc[:, :-1])
labels = np.array(data.iloc[:, -1])
labels = labels.astype(int)
label_num = len(set(labels))
num_per_label = 200
ratio = (num_per_label*label_num) / len(labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels,
                                                    test_size=ratio, random_state=20191129)
df = pd.DataFrame(np.hstack((X_test, y_test[:, np.newaxis])))
df.to_csv(write_path, index=None)
