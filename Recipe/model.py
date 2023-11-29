import torch

model = torch.load('11-27-23_model_v2.pth', map_location=torch.device('cpu'))

import requests
from PIL import Image
from torchvision import datasets, models, transforms

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(inp):
    inp = transforms_test(inp)
    input = inp.unsqueeze(0)
    with torch.no_grad():
        output = model(input)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class=torch.argmax(probabilities).item()
    return predicted_class