import torch
import torch.nn as nn
import torch.nn.functional as F

class LossUtils:
    def __init__(self, device='cpu'):
        self.device = device
    
    def __call__(self, loss_name, **kwargs):
        return getattr(self, loss_name)(**kwargs)
        
    def MSE(self, reduce, reduction):
        return nn.MSELoss(reduce=True, reduction='mean')
        
    def CE(self):
        return nn.CrossEntropyLoss()
        
