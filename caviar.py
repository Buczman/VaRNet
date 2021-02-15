import pandas as pd
import numpy as np
import torch
from loss import caviar_loss, huber_loss
from nets import CAViaR, CAViaRLightning
from utils import predict_rolling, predict_rolling_lightning, reset_params, TimeDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer


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


def caviar_prediction(index, sample_start, training_sample, testing_sample, in_model_testing_sample, memory_size,
                      epochs_per_step, batch_size, device, huber=False):
    data = pd.read_csv('./data/' + index + '.csv').set_index('Data')
    data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)

    dataset = data.loc[(data.index > sample_start)]
    dataset = dataset.iloc[:(training_sample + testing_sample + in_model_testing_sample)]

    # model = CAViaR(device=device, stateful=False, memory_size=memory_size)
    model = CAViaRLightning()



    for i in range(testing_sample):

        reset_params(model)
        dataset_iter = dataset.log_returns[i:training_sample+i]

        X = dataset_iter.values
        y = dataset_iter.shift(-1).values

        scaler = StandardScaler()
        scaler.fit(X.reshape(-1, 1))
        X = scaler.transform(X.reshape(-1, 1)).reshape(-1)
        y = scaler.transform(y.reshape(-1, 1)).reshape(-1)

        timedataset = TimeseriesDataset(X, y, memory_size)
        training_generator = DataLoader(timedataset, batch_size=batch_size, shuffle=False)

        trainer = Trainer(gpus=[0], max_epochs=epochs_per_step)
        trainer.fit(model, training_generator)

        params = model(timedataset[len(timedataset)][0].unsqueeze(1).unsqueeze(0)).cpu().detach().numpy()

        var = scaler.inverse_transform(params)
        print('VaR: %0.5f' % var[0][0])

    dataset.to_csv(
        'results/' + sample_start + 'data_caviar_' + index + '_' + str(huber) + '_' + str(memory_size) + '.csv',
        index=False)


def caviar_prediction_lightning(index, sample_start, training_sample):
    data = pd.read_csv('./data/' + index + '.csv').set_index('Data')
    data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)

    dataset = data.loc[(data.index > sample_start)]
    dataset = dataset.iloc[:(training_sample + testing_sample)]


if __name__ == "__main__":
    training_sample = 250
    testing_sample = 20
    memory_size = 10
    epochs_per_step = 400
    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    caviar_prediction('wig', '2005-01-01', training_sample, testing_sample, memory_size, epochs_per_step, batch_size,
                      device)
