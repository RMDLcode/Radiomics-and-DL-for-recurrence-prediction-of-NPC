import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
        
class DCLoss(nn.Module):
    
    def __init__(self, n=1):
        super().__init__()
        self.n = n
        
    def forward(self, pred, target, data_name):

        smooth = 1

        dice = 0.

        for i in range(self.n, pred.size(1)):
            dice += ((pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)) / ((pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)+
            0.5 * (pred[:,i] * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.5 * ((1 - pred[:,i]) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)) + smooth

        dice = dice / (pred.size(1) - self.n)
        
        target = torch.argmax(target,dim=1)
        pred = torch.log(pred+1e-8)
        criteria = nn.NLLLoss()
        ce = criteria(pred, target)
        return -dice.mean()+ce