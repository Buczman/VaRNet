import numpy as np
import torch
import torch.utils.data
import skewstudent
import time
from scipy.stats import t, norm
from nets import *


class TimeDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset, start_index, end_index, history_size, target_size):
        'Initialization'
        self.dataset = dataset
        self.history_size = history_size
        self.start_index = start_index + history_size
        if end_index is None:
            self.end_index = len(dataset) - target_size
        else:
            self.end_index = end_index
        self.target_size = target_size

    def __len__(self):
        'Denotes the total number of samples'
        return self.end_index - self.start_index

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        indices = range(index, index + self.history_size)

        # Load data and get label
        X_train = np.reshape(self.dataset[indices], (self.history_size, 1))
        y_train = np.array(self.dataset[index + self.history_size + self.target_size])

        return X_train, y_train


def reset_params(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    if isinstance(model, GARCHSkewedTStudent) or isinstance(model, GARCHTStudent):
        model.df.data = torch.tensor(2.05, device=model.device)
    if isinstance(model, GARCHSkewedTStudent):
        model.skewness.data = torch.tensor(0., device=model.device)


def predict_rolling(dataset, model, memory, epochs, optimizer, loss_function, param_list, device):
    reset_params(model)
    print('RR: %0.5f' % dataset[-1])
    train(model, optimizer, loss_function, memory, dataset, epochs, device, verbose=True)
    X_pred = torch.from_numpy(np.reshape(dataset.to_numpy()[-30:], (30, 1)))
    X_train = torch.tensor(np.expand_dims(X_pred, axis=1)).float().to(device)

    params = model(X_train).cpu().detach().numpy()

    if isinstance(model, GARCHSkewedTStudent):
        dist = skewstudent.skewstudent.SkewStudent(eta=params[1], lam=params[2])
        param_list.append([params[0], params[1], params[2]])
        var = np.sqrt(params[0])*dist.ppf(0.025)
    elif isinstance(model, GARCHTStudent):
        dist = t(df=params[1])
        param_list.append([params[0], params[1]])
        var = np.sqrt(params[0])*dist.ppf(0.025)
    elif isinstance(model, GARCH):
        dist = norm()
        param_list.append([params])
        var = np.sqrt(params)*dist.ppf(0.025)
    elif isinstance(model, CAViaR):
        param_list.append([params])
        var = params
    else:
        var = 0
    print('VaR: %0.5f' % var)
    return var


def train(model, optimizer, loss_fn, memory, dataset, epochs=20, device='cuda', verbose=True):
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

    history = {}  # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}

    timedataset = TimeDataset(dataset.to_numpy(), 0, None, memory, 0)
    training_generator = torch.utils.data.DataLoader(timedataset, **params)

    start_time_sec = time.time()

    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        train_loss = 0.0



        for batch in training_generator:
            optimizer.zero_grad()

            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(y, yhat)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)

            optimizer.step()

            train_loss += loss.data.item()

            with torch.no_grad():
                if isinstance(model, GARCHSkewedTStudent) or isinstance(model, GARCHTStudent):
                    model.df.data = model.df.clamp(+2.05, +300)
                if isinstance(model, GARCHSkewedTStudent):
                    model.skewness.data = model.skewness.clamp(-1, +1)

        train_loss = train_loss / n

        if verbose:
            print('Epoch %3d /%3d, batches: %d | train loss: %5.5f' % (epoch + 1, epochs, n, train_loss))

        history['loss'].append(train_loss)


    # END OF TRAINING LOOP

    end_time_sec = time.time()
    total_time_sec = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    if verbose:
        print()
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history
