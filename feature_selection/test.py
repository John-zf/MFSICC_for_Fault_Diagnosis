import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from feature_selection.mfsicc_plot import MFSICC
from skfeature.utility.sparse_learning import construct_label_matrix
from feature_selection.RFS import rfs
import warnings
warnings.filterwarnings("ignore")

dataset = 'PHM09_Low_COMB.csv'
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

mfsicc = MFSICC(lamb1=0.1, alpha=0.5, lamb2=0.1, random_state=1.1, n_iter=30)
mfsicc.fit(features, labels, groups)
W, objs = rfs(features, construct_label_matrix(labels), gamma=0.1)
print(mfsicc.get_objectives())
