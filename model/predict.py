import argparse
import PIL
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

# Rebuild the model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 3)

# Parse image path from command line
parser = argparse.ArgumentParser(description="Skin Burn Severity Analyzer")
parser.add_argument("image", help="Path to image file")
args = parser.parse_args()

# Load the trained data (weights)
model.load_state_dict(torch.load('model.pth'))
model.eval()
  
# Preprocessing (match training normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and transform image
test_image = PIL.Image.open(args.image).convert("RGB")
input_tensor = transform(test_image).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
model = model.to(device)


with torch.no_grad():
    outputs = model(input_tensor)
    
    probs = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(outputs, dim=1).item()
    
    
class_map = {0: '1st degree', 1: '2nd degree', 2: '3rd degree'}
print("Probabilities:", probs.tolist())
print("Prediction:", class_map[predicted_class])    