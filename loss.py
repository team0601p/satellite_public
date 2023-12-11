import torch
import torch.nn as nn
import torch.nn.functional as F

# input, target size : (batch_size, 1, width, height)

class DiceLoss(nn.Module):
    def __init__(self, smooth = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, input, target):
        pred = torch.sigmoid(input)
        intersection = torch.sum(pred * target, dim = (2, 3))
        union = torch.sum(pred + target, dim = (2, 3))
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
    
        return dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, dice_smooth = 1e-5):
        super(BCEDiceLoss, self).__init__()
        self.dice = DiceLoss(dice_smooth)
    
    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='mean').mean()
        dice = self.dice(input, target)
    
        return bce + dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, device = 'cuda'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma
    
    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none').view(-1)
        target = target.type(torch.long)
        at = self.alpha.gather(0, target.view(-1))
        pt = torch.exp(-bce)
        F_loss = at*(1-pt)**self.gamma * bce
        return F_loss.mean()

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, device = 'cuda'):
        super(BCEFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma
    
    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none').view(-1)
        target = target.type(torch.long)
        at = self.alpha.gather(0, target.view(-1))
        pt = torch.exp(-bce)
        F_loss = (at*(1-pt)**self.gamma + 1) * bce
        return F_loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self):
        super(TverskyLoss, self).__init__()
        pass
    
    def forward(self, input, target):
        pass