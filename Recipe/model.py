import torch
import matplotlib.pyplot as plt
import numpy as np

model = torch.load('checkpoint_epoch_36.pth', map_location=torch.device('cpu'))

import requests
from PIL import Image
from torchvision import datasets, models, transforms

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

labels = ["BEANS", "CAKE_MIX", "COFFEE", "CORN", "FLOUR", "HONEY", "JAM", "JUICE", "Lemon", "NUTS", "OIL", "Onion", "Orange", "PASTA", "Peas", "Potato", "RICE", "SPICES", "SUGAR", "Strawberry", "TEA", "TOMATO_SAUCE", "TUNA", "VINEGAR", "WATER","Zucchini", "apples", "aubergine", "avocado", "bananas", "blueberries", "bread", "broccoli", "butter", "carrots", "cheese", "cherry", "chicken", "eggs", "ginger", "green beans", "green chilies", "ham", "kiwi", "lettuce", "lime", "mango", "milk", "mushrooms", "peach", "peppers", "pineapple", "spinach", "sweet_potato", "tomatoes", "watermelon", "yoghurt"]

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
    model.eval()
    # Process image
    inp = transforms_test(inp)
    model_input = inp.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(10)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    top_labels = []
    for i in range(len(top_probs)):
        if top_probs[i] > 20:
            top_labels.append(labels[top_labs[i]])
        else:
            break
    
    # # Convert indices to classes
    # top_labels = [labels[lab] for lab in top_labs]
    return top_probs, top_labels
