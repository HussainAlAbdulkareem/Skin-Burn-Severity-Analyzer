import torchvision.models as models
import torch.nn as nn
import torch
from load_dataset import train_dataloader, val_dataloader


model = models.resnet18(pretrained=True)


model.fc = nn.Linear(512, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


print(model.fc)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        loss = (criterion(outputs, labels))
        
        optimizer.zero_grad()
        loss.backward()
        
        
        optimizer.step()