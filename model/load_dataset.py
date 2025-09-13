import matplotlib
import matplotlib.pyplot as plt
from torch import tensor
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader


torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomRotation(15), 
    transforms.ColorJitter(0.1,0.1,0.1,0.1),
    transforms.RandomErasing(p=0.2),  
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225]
),          
])

dataset = ImageFolder(root='C:\\Users\\Hussain\\Dropbox\PC\\Desktop\\Projects\\Skin-Burn-Severity-Analyzer\\dataset', transform=transform)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])


train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)

i = 0
for images, labels in val_dataloader:
    print(labels[i])
    i+=1
