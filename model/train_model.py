import torchvision.models as models
import torch.nn as nn

model = models.resnet18(pretrained=True)


model.fc = nn.Linear(512, 3)

print(model.fc)