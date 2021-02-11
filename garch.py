import pandas as pd
import numpy as np
import torch
from loss import hansen_garch_skewed_student_loss, garch_normal_loss
from nets import GARCHSkewedTStudent, GARCH
from utils import predict_rolling


def garch_prediction(index, sample_start, training_sample, testing_sample, in_model_testing_sample,  memory_size, epochs_per_step, batch_size, device, dist, save_name=''):
    data = pd.read_csv('./data/' + index + '.csv').set_index('Data')
    data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)

    dataset = data.loc[(data.index > sample_start)]
    dataset = dataset.iloc[:(training_sample + testing_sample + in_model_testing_sample)]

    if dist == 'skewstudent':
        model = GARCHSkewedTStudent(device=device, memory_size=memory_size)
        loss_function = hansen_garch_skewed_student_loss
        optimizer = torch.optim.Adam([{'params': model.parameters()},
                                      {'params': [model.skewness, model.df], 'lr': 1e-3}], lr=3e-4)
    else:
        model = GARCH(device=device, memory_size=memory_size)
        loss_function = garch_normal_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    param_list = []
    dataset['garch_var'] = dataset.log_returns.rolling(
        training_sample + in_model_testing_sample).apply(
        predict_rolling,
        kwargs={'model': model,
                'training_size': training_sample,
                'memory': memory_size,
                'batch_size': batch_size,
                'loss_function': loss_function,
                'optimizer': optimizer,
                'epochs': epochs_per_step,
                'param_list': param_list,
                'device': device},
        raw=False).shift(
        periods=1
    )
    if save_name != '':
        pd.DataFrame(param_list).to_csv('results/' + save_name + '_params.csv', index=False)
        dataset.to_csv('results/' + save_name + '.csv')
    else:
        pd.DataFrame(param_list).to_csv('results/' + sample_start + 'data_garch_params_' + index + '_' + dist + '_' + str(memory_size) + '.csv', index=False)
        dataset.to_csv('results/' + sample_start + 'data_garch_' + index + '_' + dist + '_' + str(memory_size) + '.csv')


if __name__ == "__main__":
    training_sample = 250
    testing_sample = 250
    memory_size = 10
    epochs_per_step = 400
    batch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    garch_prediction('wig', '2005-01-01', training_sample, testing_sample, memory_size, epochs_per_step, batch_size, device, 'skewstudent', 'test_garch_run')