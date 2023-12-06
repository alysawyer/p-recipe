import torch
import matplotlib.pyplot as plt
import numpy as np

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

labels = ["apples", "aubergine", "avocado", "bananas", "beans", "blueberries", "bread", "broccoli", "butter", "cake mix", "carrots", "cheese", "cherry", "chicken", "coffee", "corn", "eggs", "flour", "ginger", "green beans", "green chilies", "ham", "honey", "jam", "juice", "kiwi", "lemon", "lettuce", "lime", "mango", "milk", "mushrooms", "nuts", "oil", "onion", "orange", "pasta", "peach", "peas", "peppers", "pineapple", "potato", "rice", "spices", "spinach", "strawberry", "sugar", "sweet potato", "tea", "tomato sauce", "tomatoes", "tuna", "vinegar", "water", "watermelon", "yogurt", "zuccini"]

def predict(inp):
    inp = transforms_test(inp)
    input = inp.unsqueeze(0)
    with torch.no_grad():
        output = model(input)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class=torch.argmax(probabilities).item()
    return labels[predicted_class]

def predict2(inp):
    inp = transforms_test(inp)
    inp = inp.unsqueeze(0)
    with torch.no_grad():
        prediction = model(inp)
        pred_class = np.argmax(prediction).item()
        return pred_class
    
def predict3(inp):
    # Process image
    inp = transforms_test(inp)
    model_input = inp.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(5)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    top_labels = [labels[lab] for lab in top_labs]
    return top_probs, top_labels
