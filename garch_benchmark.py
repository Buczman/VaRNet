import arch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


TRAINING_SAMPLE = 1000
TESTING_SAMPLE = 250


def garch_predict_rolling(dataset, testing_sample, steps_back, dist):
    scaler = StandardScaler()
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

    value_at_risk = cond_mean.values + np.sqrt(cond_var).values * q
    var_transformed = scaler.inverse_transform(value_at_risk)[0]
    print(var_transformed)
    return var_transformed



def garch_benchmark(index, sample_start, training_sample, testing_sample, steps_back, dist):
    data = pd.read_csv('./data/' + index + '.csv')
    data = data.set_index(pd.to_datetime(data['Data']))
    data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)
    dataset = data.loc[(data.index > sample_start)]
    dataset = dataset.iloc[:(training_sample + testing_sample)]
    dataset['VaR'] = dataset.log_returns.rolling(training_sample + 1).apply(garch_predict_rolling, kwargs={
        'testing_sample': testing_sample,
        'steps_back': steps_back,
        'dist': dist})
    dataset.to_csv('results/garch_bench_' + dist + '_' + sample_start + '_' + str(steps_back) + '.csv', index=False)


if __name__ == "__main__":
    for start in ["2005-01-01", "2007-01-01", "2013-01-01", "2016-01-01"]:
        for steps_back in [100]:
            for dist in ['normal', 'skewstudent']:
                garch_benchmark('wig', start, 1000, 250, steps_back, dist)
