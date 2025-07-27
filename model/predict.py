import PIL
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

# Rebuild the model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 3)

# Load the trained data (weights)

model.load_state_dict(torch.load('model.pth'))
model.eval()

img_path = "./3burrn.jpg"

test_image = PIL.Image.open(img_path) 
  
resize_transform = transforms.Resize((224, 224))
toTensor_transform = transforms.ToTensor()

test_image_resized = resize_transform(test_image)

test_img_tensor = toTensor_transform(test_image_resized)

input_tensor = test_img_tensor.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_tensor = input_tensor.to(device)
model = model.to(device)


with torch.no_grad():
    outputs = model(input_tensor)
    
    probs = torch.softmax(outputs, dim=1)
    print(probs)
    
    predicted_class = torch.argmax(outputs, dim=1).item()
    
    
class_map = {0: '1st degree', 1: '2nd degree', 2: '3rd degree'}
print("Prediction:", class_map[predicted_class])    