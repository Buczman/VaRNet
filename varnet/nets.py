import torch
import torch.nn as nn
import pytorch_lightning.core.lightning as pl
import numpy as np
import skewstudent


class VaRNet(pl.LightningModule):

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion,
                 dist):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.dist = dist

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        y_pred = self.linear2(y_pred)
        y_pred = self.linear3(y_pred)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y, y_hat)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def predict_var(self, x):
        if self.dist is None:
            return self.forward(x).detach().numpy()
        elif isinstance(self.dist(), skewstudent.skewstudent.SkewStudent):
            var = self.forward(x).detach().numpy()[0]
            return np.array([np.sqrt(var[:1]) * self.dist(eta=var[2], lam=var[1]).ppf(0.025)])
        else:
            return np.sqrt(self.forward(x).detach().numpy()) * self.dist().ppf(0.025)


class GARCHVaRNet(VaRNet):

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion,
                 dist):
        super().__init__(n_features,
                         hidden_size,
                         seq_len,
                         batch_size,
                         num_layers,
                         dropout,
                         learning_rate,
                         criterion,
                         dist)

        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        y_pred = self.linear2(y_pred)
        y_pred = self.linear3(y_pred)
        return self.softplus(y_pred)

class SkewedGARCHVaRNet(GARCHVaRNet):

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion,
                 dist):
        super().__init__(n_features,
                         hidden_size,
                         seq_len,
                         batch_size,
                         num_layers,
                         dropout,
                         learning_rate,
                         criterion,
                         dist)

        self.linear3_1 = torch.nn.Linear(32, 1)
        self.linear3_2 = torch.nn.Linear(32, 1)
        self.linear3_3 = torch.nn.Linear(32, 1)

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        y_pred = self.linear2(y_pred)

        y_pred_1 = self.softplus(self.linear3_1(y_pred))  # vol
        y_pred_2 = self.tanh(self.linear3_2(y_pred))  # skew
        y_pred_3 = self.relu(self.linear3_3(y_pred)) + 2.05  # df

        return torch.cat([
            y_pred_1,
            y_pred_2,
            y_pred_3
        ], 1)
