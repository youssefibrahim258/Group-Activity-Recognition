import torchvision.models as models
import torch.nn as nn
import torch

class ResNetB1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # all layers except final fc
        self.classifier = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
