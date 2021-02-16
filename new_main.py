from time import time
import skewstudent
import numpy as np
import pandas as pd

from loss import caviar_loss, huber_loss, hansen_garch_skewed_student_loss

pd.options.display.float_format = '{:,.5f}'.format

# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Neural Networks
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers.csv_logs import CSVLogger

# Plotting

import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


class MyPrintingCallback(Callback):

    def on_train_end(self, trainer, pl_module):
        pl_module.forward()


class TimeseriesDataset(Dataset):
    '''
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    '''

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len)

    def __getitem__(self, index):
        return (self.X[index:index + self.seq_len], self.y[index + self.seq_len - 1])


class PowerConsumptionDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    '''

    def __init__(self, df, start, test_case, seq_len=1, batch_size=128, num_workers=0):
        super().__init__()
        self.df = df
        self.start = start
        self.test_case = test_case
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.preprocessing = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        '''
        Data is resampled to hourly intervals.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        '''

        if stage == 'fit' and self.X_train is not None:
            return
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return

        self.df = self.df.iloc[0 + self.test_case:]
        self.df = self.df.iloc[:1001]
        print(len(self.df))

        X = self.df[['log_returns']].values[:1000]
        y = self.df[['log_returns']].shift(-1).values[:1000]

        self.preprocessing = StandardScaler()
        self.preprocessing.fit(X)

        self.X_test = self.df[['log_returns']].values[-self.seq_len - 1:-1]
        self.X_test = self.preprocessing.transform(self.X_test)



        if stage == 'fit' or stage is None:
            self.X_train = self.preprocessing.transform(X)
            self.y_train = self.preprocessing.transform(y)
            # self.X_val = preprocessing.transform(X_val)
            # self.y_val = y_val.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.X_test = self.preprocessing.transform(X_test)
            self.y_test = self.preprocessing.transform(y_test)

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train,
                                          self.y_train,
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    # def val_dataloader(self):
    #     val_dataset = TimeseriesDataset(self.X_val,
    #                                     self.y_val,
    #                                     seq_len=self.seq_len)
    #     val_loader = DataLoader(val_dataset,
    #                             batch_size=self.batch_size,
    #                             shuffle=False,
    #                             num_workers=self.num_workers)
    #
    #     return val_loader

    # def test_dataloader(self):
    #     test_dataset = TimeseriesDataset(self.X_test,
    #                                      self.y_test,
    #                                      seq_len=self.seq_len)
    #     test_loader = DataLoader(test_dataset,
    #                              batch_size=self.batch_size,
    #                              shuffle=False,
    #                              num_workers=self.num_workers)
    #
    #     return test_loader


class GARCHLSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion):
        super(GARCHLSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3_1 = torch.nn.Linear(32, 1)
        self.linear3_2 = torch.nn.Linear(32, 1)
        self.linear3_3 = torch.nn.Linear(32, 1)

        self.softplus = torch.nn.Softplus()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        y_pred = self.linear2(y_pred)

        y_pred_1 = self.softplus(self.linear3_1(y_pred))     # vol
        y_pred_2 = self.tanh(self.linear3_2(y_pred))         # skew
        y_pred_3 = self.relu(self.linear3_3(y_pred)) + 2.05  # df

        return torch.cat([
            y_pred_1,
            y_pred_2,
            y_pred_3
        ], 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y, y_hat)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y)
    #     self.log('val_loss', loss, on_epoch=True)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y)
    #     self.log('y_hat', y_hat, on_epoch=True)
    #     return loss


p = dict(
    start='2016-01-01',
    seq_len=100,
    batch_size=512,
    criterion=hansen_garch_skewed_student_loss,
    max_epochs=300,
    n_features=1,
    hidden_size=100,
    num_layers=1,
    dropout=0.2,
    learning_rate=3e-4,
)

seed_everything(1)

path = './data/wig.csv'
data = pd.read_csv(
    path,
    sep=',',
    index_col='Data'
)
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)
data = data.loc[(data.index > '2016-01-01')]

var_output = []
num_train = 250

for test_case in range(num_train):
    csv_logger = CSVLogger('./', name='lstm', version=str(test_case))
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=csv_logger,
        gpus=1,
        progress_bar_refresh_rate=2,
    )

    model = GARCHLSTMRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        seq_len=p['seq_len'],
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate']
    )
    dm = PowerConsumptionDataModule(
        df=data,
        start=p['start'],
        test_case=test_case,
        seq_len=p['seq_len'],
        batch_size=p['batch_size'],
    )

    trainer.fit(model, datamodule=dm)
    model.eval()
    # train_test = TimeseriesDataset(dm.X_train, dm.y_train, 10)
    # pred_out = []
    # true_out = []
    # for i in range(len(train_test)):
    #     pred_out.append(model.forward(torch.tensor(train_test[i][0], dtype=torch.float).unsqueeze(0)).data)
    #     true_out.append(train_test[i][1].data)
    #
    # plt.plot(pred_out)
    # plt.plot(true_out)
    # plt.show()

    # train_test = TimeseriesDataset(dm.X_train, dm.y_train, 10)
    # pred_out = []
    # true_out = []
    # for i in range(len(train_test)):
    #     VaR = model.forward(torch.tensor(train_test[i][0], dtype=torch.float).unsqueeze(0)).detach().numpy()[0]
    #     dist = skewstudent.skewstudent.SkewStudent(eta=VaR[2], lam=VaR[1])
    #     var = np.sqrt(VaR[0]) * dist.ppf(0.025)
    #     pred_out.append(var)
    #     true_out.append(train_test[i][1].data)
    #
    # plt.plot(pred_out)
    # plt.plot(true_out)
    # plt.show()


    # VaR = model.forward(torch.tensor(dm.X_test, dtype=torch.float).unsqueeze(0))
    # print(dm.preprocessing.inverse_transform(VaR.data))
    # var_output.append(dm.preprocessing.inverse_transform(VaR.data)[0][0])

    VaR = model.forward(torch.tensor(dm.X_test, dtype=torch.float).unsqueeze(0)).detach().numpy()[0]
    dist = skewstudent.skewstudent.SkewStudent(eta=VaR[2], lam=VaR[1])
    # param_list.append([params[0], params[1], params[2]])
    var = np.sqrt(VaR[0] * dm.preprocessing.scale_ ** 2) * dist.ppf(0.025)
    print(VaR)
    print(var)
    var_output.append(var)

data.iloc[:1000+num_train].reset_index().join(pd.DataFrame(var_output, index=list(range(1000, 1000+num_train)), columns=['VaR']))[['Data', 'log_returns', 'VaR']].to_csv('test_garch_100.csv')

