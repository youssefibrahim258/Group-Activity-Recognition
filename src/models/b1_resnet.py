import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNetB1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base_model = resnet50(pretrained=True)

        for p in base_model.parameters():
            p.requires_grad = False

        for p in base_model.layer3.parameters():
            p.requires_grad = True

        for p in base_model.layer4.parameters():
            p.requires_grad = True

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)          
        x = torch.flatten(x, 1)       
        return self.classifier(x)
