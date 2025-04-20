import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        N = target.size(0)  # 1
        C = target.size(1)  # 1
        smooth = 0.0001
        input_flat = input.view(N, C, -1)
        target_flat = target.view(N, C, -1)
        intersection = input_flat * target_flat
        loss = (2 * intersection.sum(2) + smooth) / (input_flat.sum(2) + target_flat.sum(2) + smooth)
        loss = 1 - loss.sum() / N
        return loss