import numpy as np
import torch
import skewstudent
import time
from scipy.stats import t, norm
from nets import *


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        X_train = torch.from_numpy(np.expand_dims(np.reshape(dataset[indices], (history_size, 1)), axis=1)).float()
        yield X_train, torch.from_numpy(np.array(dataset[i + target_size])).float()


def reset_params(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    if isinstance(model, GARCHSkewedTStudent) or isinstance(model, GARCHTStudent):
        model.df.data = torch.tensor(2.05)
    if isinstance(model, GARCHSkewedTStudent):
        model.skewness.data = torch.tensor(0.)


def predict_rolling(dataset, model, epochs, optimizer, loss_function, param_list, device):
    reset_params(model)
    train(model, optimizer, loss_function, dataset, epochs, device, verbose=True)
    X_pred = torch.from_numpy(np.reshape(dataset.to_numpy()[-30:], (30, 1)))
    X_train = torch.tensor(np.expand_dims(X_pred, axis=1)).float().to(device)

    params = model(X_train).detach().numpy()

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


def train(model, optimizer, loss_fn, dataset, epochs=20, device='cuda', verbose=True):
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

    start_time_sec = time.time()

    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        train_loss = 0.0

        for n, batch in enumerate(univariate_data(dataset.to_numpy(), 0, None, 30, 0)):
            optimizer.zero_grad()

            x = batch[0].to(device)
            y = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(y, yhat)

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * x.size(0)

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
