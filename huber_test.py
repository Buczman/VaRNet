import torch
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-0.1, 0.1, 0.0001)


def caviar_loss(x):
    return -1*(float(x < 0) - 0.025) * x

def caviar_loss_2(x):
    return -1*(float(x < 0) - 0.025) * huber_loss(x) * torch.sign(x)


def huber_loss(x, eps=0.025):
    if torch.abs(x) <= eps:
        return torch.square(x) / 2
    else:
        return eps * (torch.abs(x) - 1 / 2 * eps)


pred = []
pred2 = []
# for i in range(len(x)):
#     pred.append(caviar_loss_2(torch.from_numpy(x)[i]))
#     pred2.append(caviar_loss(torch.from_numpy(x)[i]))

for i in range(len(x)):
    pred.append(torch.from_numpy(x)[i])
    pred2.append(1/0.025*huber_loss(torch.from_numpy(x)[i])*torch.sign(torch.from_numpy(x)[i]))

plt.plot(x, pred)
plt.plot(x, pred2)
plt.show()

