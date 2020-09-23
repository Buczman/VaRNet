import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

data = pd.read_csv('./data/wig.csv').set_index('Data')
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1]/x[0]), raw=True)


dataset = data.loc[(data.index > '2005-01-01')]
dataset2 = dataset.iloc[1:501]

history_size = 30

def univariate_data(dataset, start_index, end_index, history_size, target_size):

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        variance = np.zeros((history_size, 1))
        indices = range(i-history_size, i)
        variance[-1, :] = np.var(dataset[indices])
        X_train = np.reshape(dataset[indices], (history_size, 1))
        X_train = np.concatenate((X_train, variance), axis=1)
        # Reshape data from (history_size,) to (history_size, 1)
        yield X_train, np.array(dataset[i+target_size])


def garch_loss(true, vol):
    return 1 / 2 * (torch.log(vol) + true ** 2 / vol)  #    + tf.math.log(2 * tf.constant(np.pi))


class LSTM(torch.nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)

        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.sigmoid(self.linear(lstm_out.view(len(input_seq), -1)))
        return predictions[-1]

model = LSTM()
loss_function = garch_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




epochs = 15


list_of_outputs = []

for i in range(epochs):
    for seq, labels in univariate_data(dataset2.log_returns.to_numpy(), 0, None, 30, 0):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        if len(list_of_outputs) < history_size:
            seq[history_size - len(list_of_outputs):history_size, 1] = list_of_outputs
            X_train = torch.tensor(np.expand_dims(seq, axis=1)).float()
            y_pred = model(X_train)
            list_of_outputs.append(y_pred.data.numpy()[0])
        else:
            X_train = torch.tensor(np.expand_dims(seq, axis = 1)).float()
            y_pred = model(X_train)
            del list_of_outputs[0]
            list_of_outputs.append(y_pred.data.numpy()[0])
        single_loss = loss_function(torch.tensor(labels).float(), y_pred)
        single_loss.backward()
        optimizer.step()

    print(f'epoch: {i:3} loss: {single_loss.tolist()[0]:10.8f}')

    if i == epochs-1:
        print(torch.sqrt(y_pred) * scipy.stats.norm.ppf(0.025))






# dataset['var'] = dataset.log_returns.rolling(251).apply(predict_rolling, raw=True)
# dataset.to_csv('./data_garch.csv')