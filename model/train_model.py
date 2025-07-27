import torchvision.models as models
import torch.nn as nn
import torch

model = models.resnet18(pretrained=True)


model.fc = nn.Linear(512, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


print(model.fc)