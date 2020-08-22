# -*- coding: utf-8 -*-
"""
Created on 2019/11/21 11:34

@author: John_Fengz
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
torch.manual_seed(20191121)


# Hyper parameters
EPOCH = 500
BATCH_SIZE = 64
LR = 0.0002
EMBED_DIM = 100

file_path = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Features\\' \
                 'Datasets_Sample200\\PHM09_Low_Raw.csv'
write_path = 'F:\\MyResearch\\PHM\\Experiments\\2 GroupL21\\Features\\' \
                 'SAE\\PHM09_Low_SAE.csv'
df = pd.read_csv(file_path)
data = np.array(df.iloc[:, :-1])
labels = np.array(df.iloc[:, -1])
data = scale(data)
data = torch.from_numpy(data).float()
deal_data = TensorDataset(data)
train_loader = DataLoader(dataset=deal_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, EMBED_DIM),
        )

        self.decoder = nn.Sequential(
            nn.Linear(EMBED_DIM, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()
optimizer = optim.RMSprop(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, batch in enumerate(train_loader):
        x_batch = batch[0].view(-1, 1024)
        y_batch = batch[0].view(-1, 1024)
        encoded_, decoded_ = autoencoder(x_batch)
        loss = loss_func(decoded_, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('EPOCH-{} Batch-{} Loss-{}'.format(epoch, step, loss.data.numpy()))


features, _ = autoencoder(data)
feature_out = features.data.numpy()
data_out = np.hstack((feature_out, labels[:, np.newaxis]))
df = pd.DataFrame(data_out)
df.to_csv(write_path, index=None)
