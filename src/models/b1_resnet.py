from torch import nn
from torchvision import models


class ResNetB1(nn.Module):
    def __init__(self, num_classes=8,freeze_backbone=True):
        super(ResNetB1, self).__init__()
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        self.in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(nn.Dropout(p=0.3),nn.Linear(self.in_features, num_classes))

    def forward(self, x):
        return self.model(x)