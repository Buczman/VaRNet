import numpy as np
import pandas as pd
from scipy.optimize import minimize


def caviar(data):

    VaR = np.zeros(len(data))

    emp_qnt = np.quantile(data, 0.025) if len(data) < 300 else np.quantile(data[:300], 0.025)

    starting_params = np.random.random(3)

    res = minimize(loss, starting_params, args=(data, emp_qnt, VaR), method='L-BFGS-B',
             bounds=((0, None), (0, None), (0, None)))

    params = res.x

    return -1 * np.sqrt(params[0] + params[1] * np.square(VaR[-1]) + params[2] * np.square(data[-1]))



def loss(params, data, emp_qnt, VaR, pval=0.025):

    VaR[0] = emp_qnt

    for i in range(1, len(data)):
        VaR[i] = -1 * np.sqrt(params[0] + params[1] * np.square(VaR[i - 1]) + params[2] * np.square(data[i-1]))

    hit = (data < VaR) - pval
    RQ = -1 * np.sum(np.dot(hit, (data - VaR)))

    return RQ



data = pd.read_csv('./data/wig.csv')
data = data.set_index(pd.to_datetime(data['Data']))
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)


dataset = data.loc[(data.index > '2005-01-01')]
dataset = dataset.iloc[:250+250]

dataset['caviar'] = dataset.log_returns.rolling(250).apply(caviar, raw=True).shift(1)
dataset.to_csv('results/data_benchmark_caviar.csv', index=False)

