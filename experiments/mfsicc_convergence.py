# -*- coding: utf-8 -*-
"""
# Created on 2020/04/22 11:51
# @author: 张峰

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from feature_selection.mfsicc_plot import MFSICC
import warnings
warnings.filterwarnings("ignore")

val = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
with open('H:\\StudyAndResearch\\MFSICC\\plot\\convergence\\1.csv', 'w') as w:
	for i in val:
		print(i)
		random_state = i
		dataset = 'CWRU_HP0_COMB.csv'
		'''
		CWRU_HP1_COMB.csv
		CWRU_HP2_COMB.csv
		CWRU_HP3_COMB.csv
		PHM09_High_COMB.csv
		PHM09_Low_COMB.csv
		'''
		file_group = '..//features//Groups.csv'
		groups = np.array(pd.read_csv(file_group))[:, 0]

		file_path = '..//features//' + dataset
		data = pd.read_csv(file_path)
		features = np.array(data.iloc[:, :-1])
		features = scale(features)
		labels = np.array(data.iloc[:, -1])
		labels = labels.astype(int)

		mfsicc = MFSICC(lamb1=0.1, alpha=0.5, lamb2=0.1, random_state=random_state, n_iter=30)
		mfsicc.fit(features, labels, groups)
		values = [str(x) for x in mfsicc.get_objectives()/10.0]
		w.write(','.join(values) + '\n')
