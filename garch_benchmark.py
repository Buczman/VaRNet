import arch
import pandas as pd
import numpy as np

TRAINING_SAMPLE = 250
TESTING_SAMPLE = 250


data = pd.read_csv('./data/wig.csv')
data = data.set_index(pd.to_datetime(data['Data']))
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)


dataset = data.loc[(data.index > '2005-01-01')]
dataset = dataset.iloc[:(TRAINING_SAMPLE + TESTING_SAMPLE)]




def predict_rolling(dataset):
    am = arch.arch_model(dataset, mean='Zero', vol='Garch', p=30, o=0, q=30, dist='skewstudent')
    res = am.fit(disp='off', last_obs=251)


    forecasts = res.forecast()
    cond_mean = forecasts.mean.iloc[-1]
    cond_var = forecasts.variance.iloc[-1]

    q = am.distribution.ppf([0.025], res.params[-2:])

    value_at_risk = cond_mean.values + np.sqrt(cond_var).values * q

    return value_at_risk[0]

dataset['garch_var'] = dataset.log_returns.rolling(TRAINING_SAMPLE + 1).apply(predict_rolling)
dataset.to_csv('results/data_garch_real.csv')