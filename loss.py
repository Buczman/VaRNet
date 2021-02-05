import torch
import numpy as np


def caviar_loss(true, var, pval=0.01):
    return torch.mean(-1*((true < var).float() - pval) * (true - var))

def huber_loss(true, var, pval=torch.tensor(0.025), eps=torch.tensor(0.025)):
    x = true - var
    return torch.mean(torch.cat([
        x[x <= (pval - 1) * eps] * (pval - 1) - 1 / 2 * (pval - 1) ** 2 * eps,
        x[(x > (pval - 1) * eps) & (x <= pval * eps)] ** 2 / (2 * eps),
        x[x > pval * eps] * pval - 1 / 2 * pval ** 2 * eps
    ]))

def garch_normal_loss(true, vol):
    return 1 / 2 * torch.mean(torch.log(vol) + true ** 2 / vol)  # + tf.math.log(2 * tf.constant(np.pi))


def student_loss(true, pred):
    vol = pred[0]
    df = pred[1] + 2

    llh = + 1/2 * (
        torch.log(vol) + (1+df)*torch.log(1 + torch.square(true)/(vol * (df - 2)))
    )

    return llh


def hansen_garch_skewed_student_loss(true, pred):

    vol = pred[:-2]
    df = pred[-2]
    skewness = pred[-1]

    c = torch.lgamma((df + 1)/2) - torch.lgamma(df / 2) - torch.log(np.pi * (df - 2)) / 2

    a = 4 * skewness * torch.exp(c) * (df-2) / (df-1)

    b = torch.sqrt(1 + 3*torch.square(skewness) - torch.square(a))

    z = true / torch.sqrt(vol)

    lls = torch.log(b) + c - torch.log(vol) / 2

    llf_resid = torch.square((b * z + a) / (1 + torch.sign(z + a / b) * skewness))

    lls -= (df + 1) / 2 * torch.log(1 + llf_resid / (df - 2))

    lls *= -1

    return torch.mean(lls)