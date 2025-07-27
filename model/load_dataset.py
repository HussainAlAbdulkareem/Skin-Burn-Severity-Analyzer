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
    transforms.ToTensor(),
    
])

dataset = ImageFolder(root='Skin-Burn-Severity-Analyzer\\dataset', transform=transform)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size], seed=42)


train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)


print(train_dataloader)
print(val_dataloader)

# t = dataset[0][0]

# t_numpy = t.numpy()


# transposed_arr = np.transpose(t_numpy, axes=(1, 2, 0))

# plt.imshow(transposed_arr)
# plt.show()