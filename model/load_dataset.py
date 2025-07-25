from torchvision import transforms
from torchvision.datasets import ImageFolder


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = ImageFolder(root='Skin-Burn-Severity-Analyzer\dataset', transform=transform)



print(f"Total samples: {len(dataset)}")
print(f"Classes: {dataset.classes}")
print(f"Class to index map: {dataset.class_to_idx}")