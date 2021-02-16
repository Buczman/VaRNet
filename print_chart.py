import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

start_year = '2016-01-01'
mem_size = '10'
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
    # 'CAViaRNet Huber': 'results/{}data_caviar_{}_True_{}.csv'.format(start_year, index, mem_size),
    # 'CAViaRNet NoHuber': 'results/{}data_caviar_{}_False_{}.csv'.format(start_year, index, mem_size),
    # 'CAViaR': 'results/{}data_caviar_bench_{}_.csv'.format(start_year, index, mem_size),
    # 'GARCH normal': 'results/done/{}data_garch_bench_{}_normal_{}.csv'.format(start_year, index, mem_size),
    # 'GARCH skewstudent': 'results/{}data_garch_bench_{}_skewstudent_{}.csv'.format(start_year, index, mem_size),
    # 'GARCHNet normal': 'results/{}data_garch_{}_normal_{}.csv'.format(start_year, index, mem_size),
    # 'GARCHNet skewstudent': 'results/{}data_garch_{}_skewstudent_{}.csv'.format(start_year, index, mem_size),
    # 'GARCHNet skewstudent': 'results/test_garch_run.csv',
   "etst": "test_garch.csv"

}

labels = list(files_to_read.keys())

for n, file in enumerate(files_to_read.values()):
    data_tmp = pd.read_csv(file)
    # data_tmp = data_tmp.iloc[-testing_sample:].loc[:, ['var' in x for x in data_tmp.columns]]

    data = data.join(data_tmp, rsuffix='_2')

data = data[['Data', 'log_returns', 'VaR']]

data = data.iloc[-(testing_sample):]

plt.plot(pd.to_datetime(data.Data), data.log_returns, label='returns')
# for i in range(len(files_to_read)):
#     plt.plot(pd.to_datetime(data.Data), data.iloc[:, i+2], label=labels[i])
#     print(str(i), ':', data.loc[data.iloc[:, i+2] > data.iloc[:, 1]].shape[0])
plt.plot(pd.to_datetime(data.Data), data['VaR'])

plt.ylim(-0.15, 0.15)
plt.legend()
plt.show()

print(sum(data.log_returns < data['VaR']))
# print(sum(data_combined.log_returns < data_combined['caviar']))

