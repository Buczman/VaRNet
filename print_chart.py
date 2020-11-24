import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'results/data_caviar.csv')
data = data.iloc[-250:][['log_returns', 'var']]

caviar = pd.read_csv(r'results/data_benchmark_caviar.csv')
# caviar = caviar.iloc[-250:][['var']].rename(columns={'var':'caviar'})
caviar = caviar.iloc[-250:][['caviar', 'Data']]

data_combined = pd.concat((data, caviar), axis=1)

plt.plot(pd.to_datetime(data_combined.Data), data_combined.log_returns, label='returns')
plt.plot(pd.to_datetime(data_combined.Data), data_combined['var'], label='CAViaR Buczyński')
plt.plot(pd.to_datetime(data_combined.Data), data_combined['caviar'], label='CAViaR Engle')
plt.ylim(-0.075, 0.075)
plt.legend()
plt.show()

print(sum(data_combined.log_returns < data_combined['var']))
print(sum(data_combined.log_returns < data_combined['caviar']))