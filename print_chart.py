import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

start_year = '2016-01-01'
mem_size = '5'
index = 'wig'
training_sample = 1000
testing_sample = 250

data = pd.read_csv('./data/{}.csv'.format(index)).set_index('Data')
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1] / x[0]), raw=True)
data = data.loc[(data.index > start_year)]
data = data.iloc[:(training_sample + testing_sample)]
data = data.reset_index()
data = data[['Data', 'log_returns']]



files_to_read = {
    # 'CAViaRNet Huber': 'results/caviar_huber_{}_{}.csv'.format(start_year, mem_size),
    # 'CAViaR': 'results/caviar_bench_{}_1.csv'.format(start_year),
    'GARCH normal': 'results/garch_bench_normal_{}_{}.csv'.format(start_year, mem_size),
    'GARCH skewstudent': 'results/garch_bench_skewstudent_{}_{}.csv'.format(start_year, mem_size),
    # 'GARCHNet normal': 'results/garch_norm_{}_{}.csv'.format(start_year, mem_size),
    # 'GARCHNet skewstudent': 'results/garch_skew_{}_{}.csv'.format(start_year, mem_size),
}

labels = list(files_to_read.keys())

for n, name in enumerate(files_to_read.keys()):
    data_tmp = pd.read_csv(files_to_read[name])
    data_tmp = data_tmp.loc[(data_tmp['Data'] > start_year)].iloc[:(training_sample + testing_sample)].reset_index()
    data_tmp = data_tmp.loc[:, ['VaR' in x for x in data_tmp.columns]]
    data_tmp.columns = [name]

    data = data.join(data_tmp, rsuffix='_2')

data = data.iloc[-(testing_sample):]

plt.plot(pd.to_datetime(data.Data), data.log_returns, label='returns')
for i in range(len(files_to_read)):
    plt.plot(pd.to_datetime(data.Data), data.iloc[:, i+2], label=labels[i])
    print(str(labels[i]), ':', data.loc[data.iloc[:, i+2] > data.iloc[:, 1]].shape[0])

plt.ylim(-0.15, 0.15)
plt.legend()
plt.show()

