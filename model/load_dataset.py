import matplotlib
import matplotlib.pyplot as plt
from torch import tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np


transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    
])

dataset = ImageFolder(root='Skin-Burn-Severity-Analyzer\\dataset', transform=transform)

t = dataset[0][0]

t_numpy = t.numpy()


transposed_arr = np.transpose(t_numpy, axes=(1, 2, 0))

plt.imshow(transposed_arr)
plt.show()