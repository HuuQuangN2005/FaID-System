import torch.nn as nn
import torchvision.models as models

class LandmarkModel(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, 10)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x.view(-1,5,2)