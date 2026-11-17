"""MobileNetV3-Small encoder with a depthwise separable U-Net decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from .constants import CLASSES, MEAN, STD


class Separable(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.project = nn.Conv2d(cin, cout, 1, bias=False)
        self.fuse = nn.Sequential(Separable(cout + skip, cout), Separable(cout, cout))

    def forward(self, x, skip):
        x = F.interpolate(self.project(x), size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat((x, skip), dim=1))


class LaneUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features
        self.up = nn.ModuleList([Up(576, 48, 96), Up(96, 24, 64),
                                 Up(64, 16, 32), Up(32, 16, 16)])
        self.head = nn.Conv2d(16, len(CLASSES), 1)

    def forward(self, x):
        size = x.shape[-2:]
        skips = []
        z = x
        for i, layer in enumerate(self.encoder):
            z = layer(z)
            if i in (0, 1, 3, 8):
                skips.append(z)
        for layer, skip in zip(self.up, reversed(skips)):
            z = layer(z, skip)
        return F.interpolate(self.head(z), size=size, mode='bilinear', align_corners=False)


def load_checkpoint(path, device='cpu'):
    # Load only checkpoints that you created or trust (PyTorch pickle format).
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if tuple(checkpoint.get('classes', ())) != CLASSES:
        raise ValueError('Checkpoint class mapping does not match the lane model')
    model = LaneUNet().to(device)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint
