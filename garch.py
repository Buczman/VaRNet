import pandas as pd
import numpy as np
import torch
import skewstudent
from loss import hansen_garch_skewed_student_loss, caviar_loss
from nets import GARCHSkewedTStudent
from utils import predict_rolling



data = pd.read_csv('./data/wig.csv').set_index('Data')
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)

dataset = data.loc[(data.index > '2005-01-01')]
dataset2 = dataset.iloc[1:501]

history_size = 30



model = GARCHSkewedTStudent()
loss_function = hansen_garch_skewed_student_loss
optimizer = torch.optim.Adam([{'params': model.parameters()},
                              {'params': [model.skewness, model.df], 'lr': 1e-2}], lr=3e-4)


epochs = 45

hacky_list = []




dataset2[['var']] = dataset2.log_returns.rolling(251).apply(predict_rolling, kwargs={'model': model,
                                                                                     'loss_function': loss_function,
                                                                                     'optimizer': optimizer,
                                                                                     'epochs': epochs})
pd.DataFrame(hacky_list).to_csv('./data_garch_sst_params.csv')
dataset2.to_csv('./data_garch_sst.csv')
