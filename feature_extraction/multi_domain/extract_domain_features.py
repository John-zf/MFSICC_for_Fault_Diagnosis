# -*- coding: utf-8 -*-
"""
Created on 2019/11/21 10:08

@author: John_Fengz
"""
import pandas as pd
import numpy as np
from feature_extraction.multi_domain import multi_domain_features


def etra_fea_from_csv(file_path, write_path):
    df = pd.read_csv(file_path)
    data = np.array(df.iloc[:, :-1])
    labels = np.array(df.iloc[:, -1])
    features = []
    for k, ins in enumerate(data):
        features.append(multi_domain_features.feature_extraction(ins, 200000/3))
        print(k)
    features = np.array(features)
    data_conc = np.hstack((features, labels[:, np.newaxis]))
    df_save = pd.DataFrame(data_conc)
    df_save.to_csv(write_path, index=None)


if __name__ == '__main__':
    file_path1 = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Features\\' \
                 'Datasets_Sample200\\PHM09_High_Raw.csv'
    write_path1 = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Features\\' \
                  'Multi-Domain\\PHM09_High_Domain.csv'
    etra_fea_from_csv(file_path1, write_path1)
