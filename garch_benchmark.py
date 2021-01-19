import arch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


TRAINING_SAMPLE = 1000
TESTING_SAMPLE = 250


def garch_predict_rolling(dataset, testing_sample, steps_back, dist):
    scaler = MinMaxScaler((-10, 10))
    scaler.fit(dataset.values.reshape(-1, 1))
    dataset = scaler.transform(dataset.values.reshape(-1, 1))
    am = arch.arch_model(dataset, mean='Zero', vol='Garch', p=steps_back, o=0, q=steps_back, dist=dist)
    res = am.fit(disp='off', last_obs=testing_sample + 1)

    forecasts = res.forecast()
    cond_mean = forecasts.mean.iloc[-1]
    cond_var = forecasts.variance.iloc[-1]
    if dist == 'skewstudent':
        q = am.distribution.ppf([0.025], res.params[-2:])
    else:
        q = am.distribution.ppf([0.025])

    value_at_risk = cond_mean.values/scaler.scale_ + np.sqrt(cond_var/scaler.scale_**2).values * q

    return value_at_risk[0]



def garch_benchmark(index, sample_start, training_sample, testing_sample, steps_back, dist):
    data = pd.read_csv('./data/' + index + '.csv')
    data = data.set_index(pd.to_datetime(data['Data']))
    data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)
    dataset = data.loc[(data.index > sample_start)]
    dataset = dataset.iloc[:(training_sample + testing_sample)]
    dataset['garch_bench_var'] = dataset.log_returns.rolling(training_sample + 1).apply(garch_predict_rolling, kwargs={
        'testing_sample': testing_sample,
        'steps_back': steps_back,
        'dist': dist})
    dataset.to_csv('results/' + sample_start + 'data_garch_bench_' + index + '_' + dist + '_' + str(steps_back) + '.csv', index=False)


