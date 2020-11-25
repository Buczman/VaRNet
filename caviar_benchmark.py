import numpy as np
import pandas as pd
from scipy.optimize import minimize


def caviar(data, steps_back=1):

    VaR = np.zeros(len(data))

    if len(data > 300 + steps_back - 1):
        emp_qnt = data.iloc[:300+steps_back-1].rolling(300).quantile(quantile=0.025).iloc[-steps_back:].values
    else:
        emp_qnt = np.repeat(np.quantile(data.values, 0.025), steps_back)

    starting_params = np.random.random((steps_back * 2 + 1))

    res = minimize(loss, starting_params, args=(steps_back, data.values, emp_qnt, VaR), method='L-BFGS-B',
             bounds=((0, None), (0, None)) * steps_back + ((0, None),))

    params = res.x
    print(-1 * np.sqrt(params[0] + np.dot(params[1:steps_back+1], np.square(VaR[- (steps_back + 1):-1])) + np.dot(params[(steps_back + 1):(2*steps_back + 1)], np.square(data[-(steps_back + 1):-1].values))))
    return -1 * np.sqrt(params[0] + np.dot(params[1:steps_back+1], np.square(VaR[- (steps_back + 1):-1])) + np.dot(params[(steps_back + 1):(2*steps_back + 1)], np.square(data[-(steps_back + 1):-1].values)))


def loss(params, steps_back, data, emp_qnt, VaR, pval=0.025):

    VaR[:steps_back] = emp_qnt

    for i in range(steps_back + 1, len(data)):
        VaR[i] = -1 * np.sqrt(params[0] + np.dot(params[1:steps_back+1], np.square(VaR[i - (steps_back+1):i-1])) + np.dot(params[(steps_back+1):(2*steps_back+1)], np.square(data[i - (steps_back+1):i-1])))

    hit = (data < VaR) - pval
    RQ = -1 * np.sum(np.dot(hit, (data - VaR)))

    return RQ



data = pd.read_csv('./data/wig.csv')
data = data.set_index(pd.to_datetime(data['Data']))
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)


dataset = data.loc[(data.index > '2005-01-01')]
dataset = dataset.iloc[:1000+250]

dataset['caviar'] = dataset.log_returns.rolling(1000).apply(caviar).shift(1)
dataset.to_csv('results/data_benchmark_caviar.csv', index=False)

