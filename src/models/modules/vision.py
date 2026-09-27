import torch
import torch.nn as nn
from torchvision import models


class ResNet18FeatureExtractor(nn.Module):
    """ImageNet ResNet-18 returning its layer2-layer4 feature maps, with arbitrary leading batch dimensions."""

    def __init__(self, frozen: bool):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if frozen:
            self.requires_grad_(False)
            self.eval()

    def train(self, mode: bool = True):
        """A frozen extractor always stays in eval mode."""
        if not any(p.requires_grad for p in self.parameters()):
            return super().train(False)
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        extra_dims = x.shape[:-3]
        if extra_dims:
            x = x.flatten(0, -4)
        x = self.layer1(self.stem(x))
        features: dict[str, torch.Tensor] = {}
        for name in ("layer2", "layer3", "layer4"):
            x = getattr(self, name)(x)
            features[name] = x
        if extra_dims:
            features = {k: v.view(*extra_dims, *v.shape[1:]) for k, v in features.items()}
        return features


class SmallConvNet(nn.Module):
    """Minimal 5-layer ConvNet: (*, 3, H, W) → (*, out_dim)."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-3]
        x = x.flatten(0, -4) if x.ndim > 4 else x

        x = self.features(x)
        x = x.flatten(1)
        x = self.head(x)

        if leading:
            x = x.view(*leading, -1)
        return x
