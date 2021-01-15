import numpy as np
import torch.utils.data
import skewstudent
import time
from scipy.stats import t, norm
from nets import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR


class TimeDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dataset, start_index, end_index, history_size, target_size, device='cuda'):
        'Initialization'
        self.dataset = torch.tensor(dataset.values, device=device, dtype=torch.float)

        self.start_index = start_index + history_size
        if end_index is None:
            self.end_index = len(dataset) - target_size
        else:
            self.end_index = end_index
        self.history_size = history_size
        self.target_size = target_size

    def __len__(self):
        'Denotes the total number of samples'
        return self.end_index - self.start_index

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.dataset[index:index + self.history_size], \
               self.dataset[index + self.history_size + self.target_size:index + self.history_size + self.target_size+1]


def reset_params(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    if isinstance(model, GARCHSkewedTStudent) or isinstance(model, GARCHTStudent):
        model.df.data = torch.tensor(2.05, device=model.device)
    if isinstance(model, GARCHSkewedTStudent):
        model.skewness.data = torch.tensor(0., device=model.device)


def predict_rolling(dataset, model, memory, batch_size, epochs, optimizer, loss_function, param_list, device):
    reset_params(model)
    print('RR: %0.5f' % dataset[-1])
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
    train(model, optimizer, scheduler, loss_function, memory, batch_size, dataset, epochs, device, verbose=False)
    if not model.stateful:
        X_pred = torch.from_numpy(np.reshape(dataset.to_numpy()[-memory:], (1, memory))).float().to(device)
        params = model(X_pred).cpu().detach().numpy()
    else:
        X_pred = []
        X_train = dataset.to_numpy()[-(batch_size+memory - 1):]
        for i in range(batch_size):
            X_pred.append(torch.tensor(X_train[i:i+memory], dtype=torch.float).to(device))
        X_pred = torch.stack(X_pred)
        params = model(X_pred).cpu().detach().numpy()[-1]

    if isinstance(model, GARCHSkewedTStudent):
        dist = skewstudent.skewstudent.SkewStudent(eta=params[1], lam=params[2])
        # param_list.append([params[0], params[1], params[2]])
        var = np.sqrt(params[0]/10000) * dist.ppf(0.025)
    elif isinstance(model, GARCHTStudent):
        dist = t(df=params[1])
        # param_list.append([params[0], params[1]])
        var = np.sqrt(params[0]) * dist.ppf(0.025)
    elif isinstance(model, GARCH):
        dist = norm()
        # param_list.append([params])
        var = np.sqrt(params/10000) * dist.ppf(0.025)
    elif isinstance(model, CAViaR):
        # param_list.append([params])
        var = params
    else:
        var = 0
    print('VaR: %0.5f' % var)
    return var


def train(model, optimizer, scheduler, loss_fn, memory, batch_size, dataset, epochs=20, device='cuda', verbose=True):
    '''
    Runs training loop for classification problems. Returns Keras-style
    per-epoch history of loss and accuracy over training and validation data.

    Parameters
    ----------
    model : nn.Module
        Neural network model
    optimizer : torch.optim.Optimizer
        Search space optimizer (e.g. Adam)
    loss_fn :
        Loss function (e.g. nn.CrossEntropyLoss())
    train_dl :
        Iterable dataloader for training data.
    epochs : int
        Number of epochs to run
    device : torch.device
        Specifies 'cuda' or 'cpu'

    Returns
    -------
    Dictionary
        Similar to Keras' fit(), the output dictionary contains per-epoch
        history of training loss, training accuracy, validation loss, and
        validation accuracy.
    '''

    print('%s : train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (dataset.index[-1], type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    timedataset = TimeDataset(dataset, 0, None, memory, 0, device)
    training_generator = torch.utils.data.DataLoader(timedataset, batch_size=batch_size, shuffle=False,
                                                     drop_last=False)
    start_time_sec = time.time()

    train_loss = torch.tensor(0.0)

    for epoch in range(epochs):
        prev_train_loss = train_loss
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        train_loss = 0.0

        if model.stateful:
            model.init_hidden(batch_size)

        for n, batch in enumerate(training_generator):
            optimizer.zero_grad()

            x = batch[0]
            y = batch[1]
            yhat = model(x)
            loss = loss_fn(y, yhat)

            loss.backward()

            optimizer.step()

            train_loss += loss.data

            with torch.no_grad():
                if isinstance(model, GARCHSkewedTStudent) or isinstance(model, GARCHTStudent):
                    model.df.data = model.df.clamp(+2.05, +300)
                if isinstance(model, GARCHSkewedTStudent):
                    model.skewness.data = model.skewness.clamp(-1, +1)

        train_loss = train_loss / (n + 1)
        scheduler.step()
        # test_sigma = []
        # for i in range(200, 1, -1):
        #     X_pred = torch.from_numpy(np.reshape(dataset.values[-30 - i:-i], (1, 30))).float().to(device)
        #     params = model(X_pred).cpu().detach().numpy()
        #     test_sigma.append(params[0])
        #     # test_sigma.append(np.sqrt(params[0]) * scaler.inverse_transform(np.array(dist.ppf(0.025)).reshape(-1, 1)))
        # plt.plot(np.array(test_sigma))
        # plt.plot(dataset.values[-200:])
        # plt.show()
        if verbose:
            print('Epoch %3d /%3d, batches: %d | train loss: %5.5f' % (epoch + 1, epochs, n, train_loss))
        if isinstance(model, CAViaR):
            tol = 0.00001
        else:
            tol = 0.00001

        # if epoch > 1 and torch.abs(train_loss / prev_train_loss - 1) < tol:
        #     break


    # END OF TRAINING LOOP

    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
