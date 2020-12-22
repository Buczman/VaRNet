import pandas as pd
import numpy as np
import torch
from loss import caviar_loss
from nets import CAViaR
from utils import predict_rolling


def caviar_prediction(sample_start, training_sample, testing_sample, memory_size, epochs_per_step, batch_size, device):
    data = pd.read_csv('./data/wig.csv').set_index('Data')
    data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)

    dataset = data.loc[(data.index > sample_start)]
    dataset = dataset.iloc[:(training_sample + testing_sample)]

    model = CAViaR(device=device, stateful=False)
    loss_function = caviar_loss
    # loss_function = huber_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    param_list = []
    dataset['var'] = dataset.log_returns.rolling(
        training_sample).apply(
        predict_rolling,
        kwargs={'model': model,
                'memory': memory_size,
                'batch_size': batch_size,
                'loss_function': loss_function,
                'optimizer': optimizer,
                'epochs': epochs_per_step,
                'param_list': param_list,
                'device': device},
        raw=False).shift(periods=1)
    pd.DataFrame(param_list).to_csv('results/' + sample_start + 'data_caviar_params.csv', index=False)
    dataset.to_csv('results/' + sample_start + 'data_caviar.csv', index=False)


if __name__ == "__main__":
    training_sample = 1000
    testing_sample = 20
    memory_size = 10
    epochs_per_step = 15
    batch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    caviar_prediction(training_sample, testing_sample, memory_size, epochs_per_step, batch_size, device)
