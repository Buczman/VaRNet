import pandas as pd
import numpy as np
import torch
from loss import hansen_garch_skewed_student_loss, caviar_loss, caviar_loss_2
from nets import CAViaR
from utils import predict_rolling


TRAINING_SAMPLE = 1000
TESTING_SAMPLE = 250
MEMORY_SIZE = 30
EPOCHS_PER_STEP = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('./data/wig.csv').set_index('Data')
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)

dataset = data.loc[(data.index > '2005-01-01')]
dataset = dataset.iloc[:(TRAINING_SAMPLE + TESTING_SAMPLE)]

model = CAViaR(device=device)
# loss_function = caviar_loss(0.025)
loss_function = caviar_loss_2
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

param_list = []
dataset[['var']] = dataset.log_returns.rolling(
    TRAINING_SAMPLE).apply(
    predict_rolling,
    kwargs={'model': model,
            'loss_function': loss_function,
            'optimizer': optimizer,
            'epochs': EPOCHS_PER_STEP,
            'param_list': param_list,
            'device': device},
    raw=False).shift(
    periods=1
)
pd.DataFrame(param_list).to_csv('results/data_caviar_params.csv', index=False)
dataset.to_csv('results/data_caviar.csv', index=False)
