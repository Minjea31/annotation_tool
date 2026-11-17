import torch
from torch import nn
from torch.nn import functional as F


class DiceCrossEntropy(nn.Module):
    def __init__(self, weights=(0.2, 1.0, 2.0)):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits, target):
        logits = logits.float()
        valid = target != 255
        if not valid.any():
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, target, weight=self.weights, ignore_index=255)
        truth = F.one_hot(target.clamp(0, 2), 3).permute(0, 3, 1, 2).float()
        pred = logits.softmax(dim=1) * valid[:, None]
        truth = truth * valid[:, None]
        intersection = (pred * truth).sum((0, 2, 3))
        dice = (2 * intersection + 1.0) / (pred.sum((0, 2, 3)) + truth.sum((0, 2, 3)) + 1.0)
        return ce + (1.0 - dice[1:]).mean()


def iou_from_confusion(confusion):
    intersection = confusion.diag().double()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    ious = [float(intersection[i] / union[i]) if union[i] > 0 else None for i in range(3)]
    lanes = [x for x in ious[1:] if x is not None]
    return {'iou': ious, 'lane_miou': sum(lanes) / len(lanes) if lanes else None}
