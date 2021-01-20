import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import MinMaxScaler
from numba import jit, float32, int8


def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


def caviar(data, steps_back=1):
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(data.values.reshape(-1, 1))
    data = scaler.transform(data.values.reshape(-1, 1))
    data = data.reshape(-1).astype(np.float32)
    # data = data.values.astype(np.float32)
    if len(data > 300 + steps_back - 1):
        emp_qnt = np.percentile(strided_app(data[:300 + steps_back - 1], 300, 1), 2.5, axis=-1)[-steps_back:]
    else:
        emp_qnt = np.repeat(np.quantile(data, 0.025), steps_back)
    emp_qnt = emp_qnt.astype(np.float32)

    # starting_params = np.random.random((10000, steps_back * 2 + 1)).astype(np.float32)
    #
    # res = {}
    # for params in starting_params:
    #     res[loss_numba(params, steps_back, data, emp_qnt)] = params
    #
    # best_res = {x: y for x, y in res.items() if x in sorted(res)[:10]}
    # best_params = {}
    # for params in best_res.values():
    #     res_ = minimize(loss_numba, params, args=(steps_back, data, emp_qnt), method='Nelder-Mead')
    #     res_ = minimize(loss_numba, res_.x, args=(steps_back, data, emp_qnt), method='L-BFGS-B',
    #              bounds=((-10, 10), (-10, 10)) * steps_back + ((-10, 10),))
    #     best_params[res_.fun] = res_.x
    #
    # params = best_params[sorted(best_params)[0]]

    params = differential_evolution(loss_numba, args=(steps_back, data, emp_qnt), tol=0.000001,
                                    bounds=((-10, 10), (-10, 10)) * steps_back + ((-10, 10),)).x

    VaR = VaR_numba(params, steps_back, data, emp_qnt)
    var = -1 * np.sqrt(params[0] + params[1] * np.square(VaR[-1]) + params[2] * np.square(data[-1]))
    print(var/scaler.scale_)
    return var/scaler.scale_


@jit(nopython=False, nogil=True)
def loss_numba(params, steps_back, data, emp_qnt):
    VaR = np.empty(len(data), dtype=np.float32)
    VaR[0] = emp_qnt[0]

    for i in range(steps_back, len(data)):
        to_sqrt = params[0] + params[1] * np.square(VaR[i - 1]) + params[2] * np.square(data[i - 1])
        if to_sqrt < 0:
            return 100
        else:
            VaR[i] = -1 * np.sqrt(to_sqrt)

    hit = ((data < VaR) - 0.025).astype(np.float32)
    RQ = -1 * hit.dot(data - VaR)
    return RQ


@jit(nopython=False, nogil=True)
def VaR_numba(params, steps_back, data, emp_qnt):
    VaR = np.empty(len(data), dtype=np.float32)
    VaR[0] = emp_qnt[0]

    for i in range(steps_back, len(data)):
        to_sqrt = params[0] + params[1] * np.square(VaR[i - 1]) + params[2] * np.square(data[i - 1])
        VaR[i] = -1 * np.sqrt(to_sqrt)

    return VaR


def loss(params, steps_back, data, emp_qnt, VaR, pval=0.025):
    VaR[:steps_back] = emp_qnt

    for i in range(steps_back, len(data)):
        to_sqrt = params[0] + np.dot(params[1:steps_back + 1], np.square(VaR[i - (steps_back):i])) + np.dot(
            params[(steps_back + 1):(2 * steps_back + 1)], np.square(data[i - (steps_back):i]))
        if to_sqrt < 0:
            return 100
        else:
            VaR[i] = -1 * np.sqrt(to_sqrt)

    hit = (data < VaR) - np.array(pval, dtype=np.float32)
    RQ = -1 * np.sum(np.dot(hit, (data - VaR)))

    return RQ


def caviar_benchmark(index, sample_start, training_sample, testing_sample, steps_back):
    data = pd.read_csv('./data/' + index + '.csv')
    data = data.set_index(pd.to_datetime(data['Data']))
    data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)
    dataset = data.loc[(data.index > sample_start)]
    dataset = dataset.iloc[:(training_sample + testing_sample)]
    dataset['caviar_bench_var'] = dataset.log_returns.rolling(training_sample).apply(caviar, kwargs={
        'steps_back': steps_back}).shift(1)
    dataset.to_csv('results/' + sample_start + 'data_caviar_bench_' + index + '_' + '.csv', index=False)

