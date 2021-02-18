import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


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


class ValueAtRiskDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    '''

    def __init__(self, df, training_length=1000, seq_len=1, batch_size=128, num_workers=0):
        super().__init__()
        self.df = df
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
        self.training_length = training_length

    def prepare_data(self):
        pass


    def setup_train(self, testcase):
        data = self.df.iloc[testcase:self.training_length + testcase]

        X = data[['log_returns']].values[:self.training_length]
        y = data[['log_returns']].shift(-1).values[:self.training_length]

        self.preprocessing = StandardScaler()
        self.preprocessing.fit(X)

        self.X_train = self.preprocessing.transform(X)
        self.y_train = self.preprocessing.transform(y)

        self.X_test = data[['log_returns']].values[-self.seq_len - 1:-1]
        self.X_test = self.preprocessing.transform(self.X_test)

        self.X_test = torch.tensor(self.X_test, dtype=torch.float).unsqueeze(0)

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train,
                                          self.y_train,
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def gather_prediction(self, prediction, testcase):
        self.df.loc[self.df.index[self.training_length + testcase], 'VaR'] = prediction
        print("Date: {0} -> RR: {1} | VaR: {2}".format(*(self.df.index[self.training_length + testcase],) + tuple(
            self.df.loc[self.df.index[self.training_length + testcase], ['log_returns', 'VaR']])))
