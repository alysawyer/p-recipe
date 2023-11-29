import torch

model = torch.load('test.pth', map_location=torch.device('cpu'))

import requests
from PIL import Image
from torchvision import transforms

#list of labels
labels = []

def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences