import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='sum', ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Apply softmax to inputs to get probabilities
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Get the log probabilities
        logpt = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-logpt)
        
        # Compute the focal loss
        if self.alpha is not None:
            at = self.alpha[targets]
            logpt = logpt * at
        
        focal_loss = (1 - pt) ** self.gamma * logpt
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Example usage:
# criterion = FocalLoss(alpha=1, gamma=2)
# loss = criterion(predictions, targets)
