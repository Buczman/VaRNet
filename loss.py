import torch
import numpy as np


def caviar_loss(pval):
    def caviar_loss_pval(true, var):
        return -1*(float(true < var) - pval) * (true - var)
    return caviar_loss_pval

def caviar_loss_2(true, var):
    return (0.025 - (true < var).float()) * torch.stack([huber_loss((true - var)[i]) for i in range(len(true))])

def huber_loss(x, eps=0.025):
    if torch.abs(x) <= eps:
        return torch.square(x) / 2 * torch.sign(x)
    else:
        return eps * (torch.abs(x) - 1 / 2 * eps) * torch.sign(x)


def garch_normal_loss(true, vol):
    return 1 / 2 * (torch.log(vol) + true ** 2 / vol)  # + tf.math.log(2 * tf.constant(np.pi))


def student_loss(true, pred):
    vol = pred[0]
    df = pred[1] + 2

    llh = + 1/2 * (
        torch.log(vol) + (1+df)*torch.log(1 + torch.square(true)/(vol * (df - 2)))
    )

    return llh


def garch_skewed_student_loss(true, pred):
    """
    deprecated for now
    """
    vol = pred[0]
    df = torch.tensor(pred[1])
    skewness = torch.tensor(pred[2])

    z_t = true/torch.sqrt(vol)

    llh = - torch.lgamma((df + 1) / 2) + torch.lgamma(df / 2) + 1 / 2 * torch.log(np.pi * (df - 2)) - \
          torch.log(2 / (skewness + 1 / skewness)) - torch.log(s) + \
          1 / 2 * (torch.log(vol) + (1 + df) * torch.log(
        1 + (z_t / (df - 2)) * torch.pow(skewness, -torch.sign(true))))

    return llh


def hansen_garch_skewed_student_loss(true, pred):

    vol = pred[0]
    df = pred[1]
    skewness = pred[2]

    if torch.abs(skewness) >= 1.0:
        skewness = torch.sign(skewness) * (1.0 - 1e-6)

    c = torch.lgamma((df + 1)/2) - torch.lgamma(df / 2) - torch.log(np.pi * (df - 2)) / 2

    a = 4 * skewness * torch.exp(c) * (df-2) / (df-1)

    b = torch.sqrt(1 + 3*torch.square(skewness) - torch.square(a))

    z = true / torch.sqrt(vol)

    lls = torch.log(b) + c - torch.log(vol) / 2

    llf_resid = torch.square((b * z + a) / (1 + torch.sign(z + a / b) * skewness))

    lls -= (df + 1) / 2 * torch.log(1 + llf_resid / (df - 2))

    lls *= -1

    return lls