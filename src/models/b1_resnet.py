import torch
import torch.nn as nn
from torchvision.models import resnet50


class ExtendCNN(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()

        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNetB1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()  

        base_model = resnet50(pretrained=True)

        for p in base_model.parameters():
            p.requires_grad = False

        for p in base_model.layer4.parameters():
            p.requires_grad = True

        backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.model = ExtendCNN(backbone, num_classes)

    def forward(self, x):
        return self.model(x)
