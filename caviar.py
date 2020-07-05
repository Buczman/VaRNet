import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./data/wig.csv').set_index('Data')
data['log_returns'] = data['Zamkniecie'].rolling(2).apply(lambda x: np.log(x[1]/x[0]), raw=True)


dataset = data.loc[(data.index > '2000-01-01') & (data.index < '2009-12-31')]

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)


emp_qnt = np.array([np.quantile(dataset.log_returns, 0.025)])
input, output = univariate_data(dataset['log_returns'].values, 1200, 2200, 30, 0)

print(input.shape)

@tf.function
def caviar_loss(true, var):
    return -1*(float(true < var) - 0.025) * (true - var)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=input.shape[-2:]),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss=caviar_loss)  # tf.keras.losses.MSE

#TODO zmiana dataset generatora
train_univariate = tf.data.Dataset.from_tensor_slices((input, output))
train_univariate = train_univariate.cache().shuffle(10000).batch(256).repeat()


model.fit(train_univariate, epochs=1,
          steps_per_epoch=1e3)

input_test, output_test = univariate_data(dataset['log_returns'].values, 2200, None, 30, 0)


model.predict(input_test)
print(output_test)

plt.plot(model.predict(input_test))
plt.plot(output_test)