import torch
import torch.nn as nn
import torch.nn.functional as F


def PSNR(pred, target):
    pred = (pred) * 255
    target = (target) * 255
    mse = F.mse_loss(pred, target)
    return 20*torch.log10(255.0/torch.sqrt(mse))