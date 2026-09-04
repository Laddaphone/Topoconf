import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetBackbone(nn.Module):
    """ResNet-18 truncated at layer3 (256 -> 128 dim)."""
    def __init__(self, d_out=128):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(base.conv1.weight.mean(dim=1, keepdim=True))
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_out)

    def forward(self, x):
        h = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        h = self.layer3(self.layer2(self.layer1(h)))
        return self.fc(self.pool(h).squeeze(-1).squeeze(-1))
