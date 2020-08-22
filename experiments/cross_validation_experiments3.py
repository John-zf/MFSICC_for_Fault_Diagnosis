# -*- coding: utf-8 -*-
"""
# Created on 2020/04/14 18:37
# @author: 张峰

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import datetime
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from skfeature.function.similarity_based import lap_score
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from skfeature.utility import construct_W
from skfeature.function.statistical_based import chi_square
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Lasso
from skfeature.function.sparse_learning_based import RFS
from skfeature.utility.sparse_learning import construct_label_matrix
from skfeature.function.structure import group_fs
from feature_selection.mfsicc import MFSICC
from skfeature.utility import sparse_learning
import warnings

warnings.filterwarnings("ignore")

random_state = 12883823
dataset = 'CWRU_HP2_COMB.csv'
'''
CWRU_HP0_COMB.csv
CWRU_HP1_COMB.csv
CWRU_HP2_COMB.csv
CWRU_HP3_COMB.csv
PHM09_High_COMB.csv
PHM09_Low_COMB.csv
'''

file_path = '..//features//' + dataset
data = pd.read_csv(file_path)
features = np.array(data.iloc[:, :-1])
features = scale(features)
labels = np.array(data.iloc[:, -1])
labels = labels.astype(int)

file_group = '..//features//Groups.csv'
groups = np.array(pd.read_csv(file_group))[:, 0]

# classifier for traning the selected features
clf = SVC(gamma='auto', kernel='rbf', decision_function_shape='ovr', random_state=random_state)

# ratios of selected features
ratios = [20, 30, 40, 50, 60, 70, 80, 90]
writes = []
head = ['time', 'lap_score', 'fisher_score', 'reliefF', 'chi_square',
        'pca', 'rfe', 'random_forest', 'lasso', 'rfs', 'sgl']
for ratio_features in ratios:
	aa = str(ratio_features)
	write_path = '..//results//' + aa + '//result_' + aa + '_' + dataset
	w = open(write_path, 'w')
	w.write(','.join(head) + '\n')
	writes.append(w)

# cross validation
count = 1
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=random_state)
for train_index, test_index in rskf.split(features, labels):
	print(count)
	X_train, X_test = features[train_index], features[test_index]
	y_train, y_test = labels[train_index], labels[test_index]
	start_time = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	print(start_time)
	acc = []

	# lap_score
	method = 'lap_score'
	kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn",
	            "weight_mode": "heat_kernel", "k": 5, 't': 1}
	W = construct_W.construct_W(X_train, **kwargs_W)
	score = lap_score.lap_score(X_train, W=W)
	idx = lap_score.feature_ranking(score)
	acc_lap = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		selected_fea_train = X_train[:, idx[0:num_features]]
		selected_fea_test = X_test[:, idx[0:num_features]]
		clf.fit(selected_fea_train, y_train)
		acc_lap.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_lap)
	acc.append(acc_lap)

	# fisher_score
	score = fisher_score.fisher_score(X_train, y_train)
	idx = fisher_score.feature_ranking(score)
	acc_fish = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		selected_fea_train = X_train[:, idx[0:num_features]]
		selected_fea_test = X_test[:, idx[0:num_features]]
		clf.fit(selected_fea_train, y_train)
		acc_fish.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_fish)
	acc.append(acc_fish)

	# reliefF
	score = reliefF.reliefF(X_train, y_train)
	idx = reliefF.feature_ranking(score)
	acc_relief = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		selected_fea_train = X_train[:, idx[0:num_features]]
		selected_fea_test = X_test[:, idx[0:num_features]]
		clf.fit(selected_fea_train, y_train)
		acc_relief.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_relief)
	acc.append(acc_relief)

	# chi_square
	score = chi_square.chi_square(np.abs(X_train), y_train)
	idx = chi_square.feature_ranking(score)
	acc_cs = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		selected_fea_train = X_train[:, idx[0:num_features]]
		selected_fea_test = X_test[:, idx[0:num_features]]
		clf.fit(selected_fea_train, y_train)
		acc_cs.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_cs)
	acc.append(acc_cs)

	# pca
	acc_pca = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		pca = PCA(n_components=num_features)
		pca.fit(X_train)
		selected_fea_train = pca.transform(X_train)
		selected_fea_test = pca.transform(X_test)
		clf.fit(selected_fea_train, y_train)
		acc_pca.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_pca)
	acc.append(acc_pca)

	# rfe
	acc_rfe = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		estimator = LinearSVC(random_state=random_state)
		selector = RFE(estimator, num_features, step=1)
		selector = selector.fit(X_train, y_train)
		selected_fea_train = selector.transform(X_train)
		selected_fea_test = selector.transform(X_test)
		clf.fit(selected_fea_train, y_train)
		acc_rfe.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_rfe)
	acc.append(acc_rfe)

	# random_forest
	rf = RandomForestClassifier(n_estimators=10, random_state=random_state)
	rf.fit(X_train, y_train)
	score = rf.feature_importances_
	idx = chi_square.feature_ranking(score)
	acc_rf = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		selected_fea_train = X_train[:, idx[0:num_features]]
		selected_fea_test = X_test[:, idx[0:num_features]]
		clf.fit(selected_fea_train, y_train)
		acc_rf.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_rf)
	acc.append(acc_rf)

	# lasso
	lasso = Lasso(alpha=0.01, random_state=random_state)
	lasso.fit(X_train, y_train)
	weights = lasso.coef_.T
	idx = chi_square.feature_ranking(abs(weights))
	acc_lasso = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		selected_fea_train = X_train[:, idx[0:num_features]]
		selected_fea_test = X_test[:, idx[0:num_features]]
		clf.fit(selected_fea_train, y_train)
		acc_lasso.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_lasso)
	acc.append(acc_lasso)

	# rfs
	weights = RFS.rfs(X_train, construct_label_matrix(y_train), gamma=0.01)
	idx = sparse_learning.feature_ranking(weights)
	acc_rfs = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		selected_fea_train = X_train[:, idx[0:num_features]]
		selected_fea_test = X_test[:, idx[0:num_features]]
		clf.fit(selected_fea_train, y_train)
		acc_rfs.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_rfs)
	acc.append(acc_rfs)

	# sgl
	idx_group = np.array([[1, 16, np.sqrt(16)],
	                      [17, 28, np.sqrt(12)],
	                      [29, 60, np.sqrt(32)],
	                      [61, 160, np.sqrt(100)]]).T
	idx_group = idx_group.astype(int)
	weights, _, _ = group_fs.group_fs(X_train, y_train, 0.01, 0.01, idx_group, verbose=False)
	idx = chi_square.feature_ranking(abs(weights))
	acc_sgl = []
	for ratio_features in ratios:
		num_features = int(ratio_features / 100 * 160)
		selected_fea_train = X_train[:, idx[0:num_features]]
		selected_fea_test = X_test[:, idx[0:num_features]]
		clf.fit(selected_fea_train, y_train)
		acc_sgl.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
	print(acc_sgl)
	acc.append(acc_sgl)

	# mfsicc
	z1_list = np.array([0.01, 0.1, 1, 10, 100, 1000, 10000])
	z2_list = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
	for z1 in z1_list:
		for z2 in z2_list:
			mfsicc = MFSICC(lamb1=z1, alpha=0.5, lamb2=z2, random_state=random_state)
			mfsicc.fit(X_train, y_train, groups)
			idx = mfsicc.feature_ranking()
			acc_mfsicc = []
			for ratio_features in ratios:
				num_features = int(ratio_features / 100 * 160)
				selected_fea_train = X_train[:, idx[0:num_features]]
				selected_fea_test = X_test[:, idx[0:num_features]]
				clf.fit(selected_fea_train, y_train)
				acc_mfsicc.append(accuracy_score(y_test, clf.predict(selected_fea_test)))
			print(acc_mfsicc)
			acc.append(acc_mfsicc)

	acc = np.array(acc)
	for index, ratio_features in enumerate(ratios):
		data = list(acc[:, index])
		w = writes[index]
		line = start_time + ',' + ','.join(list(map(lambda x: str(x), data)))
		w.write(line + '\n')
		w.flush()
	count += 1

for w in writes:
	w.close()
