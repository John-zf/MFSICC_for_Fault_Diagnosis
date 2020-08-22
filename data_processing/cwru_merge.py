# -*- coding: utf-8 -*-
"""
Created on 2019/11/21 9:12

@author: John_Fengz
"""
import numpy as np
import pandas as pd
import os
import random


def merge_cwru(file_num, labels, folder_name, write_file):
    data = []
    for k, label in enumerate(labels):
        file_folder = folder_name + '\\' + label
        files = os.listdir(file_folder)
        random.shuffle(files, random=random.seed(22580))

        for file in files[0: file_num]:
            file_path = file_folder + '\\' + file
            with open(file_path, 'r') as f:
                seg = f.readlines()
                seg = [x.strip('\n') for x in seg]
                seg.append(str(k))
            data.append(seg)
        print(label)

    df = pd.DataFrame(np.array(data))
    df.to_csv(write_file, index=False)


if __name__ == '__main__':
    file_num1 = 400
    labels1 = ['Normal', 'B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021']
    folder_name1 = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Datasets\\CWRU\\HP3\\DE\\'
    write_file1 = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Features\\CWRU_Bearing\\HP3\\CWRU_HP3_Raw.csv'
    merge_cwru(file_num1, labels1, folder_name1, write_file1)
