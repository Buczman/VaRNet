import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'data_garch.csv')
data = data.iloc[-250:]

plt.plot(pd.to_datetime(data.Data), data.log_returns)
plt.plot(pd.to_datetime(data.Data), data['var'])
plt.ylim(-0.1, 0.1)
plt.show()

print(sum(data.log_returns < data['var']))