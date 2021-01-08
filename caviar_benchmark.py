import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution


def caviar(data, steps_back=1):
    VaR = np.zeros(len(data))

    if len(data > 300 + steps_back - 1):
        emp_qnt = data.iloc[:300 + steps_back - 1].rolling(300).quantile(quantile=0.025).iloc[-steps_back:].values
    else:
        emp_qnt = np.repeat(np.quantile(data.values, 0.025), steps_back)

    # starting_params = np.random.random((10000, steps_back * 2 + 1))
    #
    # res = {}
    # for params in starting_params:
    #     res[loss(params, steps_back, data.values, emp_qnt, np.zeros(len(data)))] = params
    #
    # best_res = {x: y for x, y in res.items() if x in sorted(res)[:10]}
    # best_params = {}
    # for params in best_res.values():
    #     res_ = minimize(loss, params, args=(steps_back, data.values, emp_qnt, VaR), method='Nelder-Mead')
    #     res_ = minimize(loss, res_.x, args=(steps_back, data.values, emp_qnt, VaR), method='L-BFGS-B',
    #              bounds=((0, None), (0, None)) * steps_back + ((0, None),))
    #     best_params[res_.fun] = res_.x
    #
    # params = best_params[sorted(best_params)[0]]
    params = differential_evolution(loss, args=(steps_back, data.values, emp_qnt, VaR), tol=0.00001,
                                    bounds=((-10, 10), (-10, 10)) * steps_back + ((-10, 10),)).x

    var = -1 * np.sqrt(params[0] + np.dot(params[1:steps_back + 1], np.square(VaR[- (steps_back + 1):-1])) + np.dot(
        params[(steps_back + 1):(2 * steps_back + 1)], np.square(data[-(steps_back + 1):-1].values)))
    print(var)
    return var


def loss(params, steps_back, data, emp_qnt, VaR, pval=0.025):
    VaR[:steps_back] = emp_qnt

    for i in range(steps_back + 1, len(data)):
        to_sqrt = params[0] + np.dot(params[1:steps_back + 1], np.square(VaR[i - (steps_back + 1):i - 1])) + np.dot(
            params[(steps_back + 1):(2 * steps_back + 1)], np.square(data[i - (steps_back + 1):i - 1]))
        if to_sqrt < 0:
            return 100
        else:
            VaR[i] = -1 * np.sqrt(to_sqrt)

    hit = (data < VaR) - pval
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

