import torchvision.models as models
import torch.nn as nn

num_classes = 8

class ResNetB1(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Linear(
            self.model.fc.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)

