import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Import libabries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sn
import pandas as pd
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import time
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print(torch. __version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

use_cuda = torch.cuda.is_available()
print(use_cuda)

train_dir = "/Users/jacksusank/Downloads/p-ai/fridge_data/food_data_set_training/images"

white_torch = torchvision.io.read_image("/Users/jacksusank/Downloads/p-ai/fridge_data/food_data_set_training/images/apple_pie/134.jpg")
# print (white_torch)

print("This is a picture of apple pie")
T.ToPILImage()(white_torch)









# Next, I'm going to try to make and open an image that is just the box which is labeled 

# This is the original image
from PIL import Image
# old   im = Image.open(r"/Users/jacksusank/Downloads/P-ai/fridge_data/Food_Item_Detection.v1i.yolov7pytorch/train/images/001_png_jpg.rf.07d04531a296b96d3ffad3db54a54866.jpg")
im = Image.open(r"/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train/images/asparagus_jpg.rf.8b91337853f5dd660fff7bf6c9e058b9.jpg")
width, height = im.size


# From Chat GPT

# old  label_file_path = "/Users/jacksusank/Downloads/P-ai/fridge_data/Food_Item_Detection.v1i.yolov7pytorch/train/labels/001_png_jpg.rf.07d04531a296b96d3ffad3db54a54866.txt"  # Replace with the path to your YOLO label file
label_file_path = "/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train/labels/asparagus_jpg.rf.8b91337853f5dd660fff7bf6c9e058b9.txt"

with open(label_file_path, 'r') as file:
    lines = file.readlines()


# Here I am deciphering each of the YOLO labels in this one picture and making lists to store their values


class_id_list = []
center_x_list = []
center_y_list = []
label_width_list = []
label_height_list = []
left_multiplier_list = []
right_multiplier_list = []
top_multiplier_list = []
bottom_multiplier_list = []

# This is a list of all the possible types of labels.
# old   names = ["Apple", 'Banana', 'Beans', 'Capsicum', 'Carrot', 'Cucumber', 'Curli-Flower', 'Orange', 'Tomato', 'Tomatos', 'apple', 'asparagus', 'avocado', 'banana', 'beef', 'bell_pepper', 'bento', 'blueberries', 'bottle', 'bread', 'broccoli', 'butter', 'can', 'carrot', 'cauliflower', 'cheese', 'chicken', 'chicken_breast', 'chocolate', 'coffee', 'corn', 'cucumber', 'egg', 'eggs', 'energy_drink', 'fish', 'flour', 'garlic', 'goat_cheese', 'grapes', 'grated_cheese', 'green_beans', 'ground_beef', 'guacamole', 'ham', 'heavy_cream', 'humus', 'juice', 'ketchup', 'kothmari', 'leek', 'lemon', 'lettuce', 'lime', 'mango', 'marmelade', 'mayonaise', 'milk', 'mushrooms', 'mustard', 'nuts', 'onion', 'orange', 'pak_choi', 'parsley', 'peach', 'pear', 'pineapple', 'plasticsaveholder', 'pot', 'potato', 'potatoes', 'pudding', 'red_cabbage', 'red_grapes', 'rice_ball', 'salad', 'sandwich', 'sausage', 'shrimp', 'smoothie', 'spinach', 'spring_onion', 'strawberries', 'sugar', 'sweet_potato', 'tea_a', 'tea_i', 'tomato', 'tomato_sauce', 'tortillas', 'turkey', 'watermelon', 'yogurt']
names = ["Lemon", 'Onion', 'Orange', 'Peas', 'Potato', 'Strawberry', 'Tomato', 'apples', 'aubergine', 'bananas', 'blueberries', 'bread', 'broccoli', 'butter', 'carrots', 'cheese', 'chicken', 'courgettes', 'eggs', 'ginger', 'green beans', 'green chilies', 'ham', 'lemon', 'lettuce', 'lime', 'milk', 'mushrooms', 'onion', 'orange', 'peach', 'peppers', 'potatoes', 'red onion', 'spinach', 'spring onion', 'strawberries', 'sweet_potato', 'tomatoes', 'yoghurt']


for line in lines:
    parts = line.strip().split()  # Split the line into parts

    # These are the characteristics of each of the labels
    class_id = int(parts[0])
    center_x = float(parts[1])
    center_y = float(parts[2])
    label_width = float(parts[3])
    label_height = float(parts[4])
    left_multiplier = center_x - (label_width / 2)
    right_multiplier = center_x + (label_width / 2)
    top_multiplier = center_y + (label_height / 2)
    bottom_multiplier = center_y - (label_height / 2)

 # Now I am adding each of these values to the lists I created
    class_id_list.append(class_id)
    center_x_list.append(center_x)
    center_y_list.append(center_y)
    label_width_list.append(label_width)
    label_height_list.append(label_height)
    left_multiplier_list.append(left_multiplier)
    right_multiplier_list.append(right_multiplier)
    top_multiplier_list.append(top_multiplier)
    bottom_multiplier_list.append(bottom_multiplier)
    




    # Process the extracted information as needed
    print(f"Class ID: {class_id}, Center X: {center_x}, Center Y: {center_y}, Width: {label_width}, Height: {label_height}")


    # I'm not sure why the bottom_bound is bigger than the top_bound, but I think it might be because it's being measured from the top? I might need to rename my variables.
    # In these lines I am multiplying the Yolo float values which are between 0 and 1 by the width and height of the original picture so that I get the locations of the pixels.
    print("This is a " + str(names[class_id]))
    left_bound = round(left_multiplier * width)
    print("The left bound is " + str(left_bound))
    right_bound = round(right_multiplier * width)
    print("The right bound is " + str(right_bound))
    top_bound = round(bottom_multiplier * height)
    print("The top bound is " + str(top_bound))
    bottom_bound = round(top_multiplier * height)
    print("The bottom bound is " + str(bottom_bound))


    # These lines display the items that are labeled!
    im1 = im.crop((left_bound, top_bound, right_bound, bottom_bound))
    im1.show()













# This block of code will display the original image that goes along with these labels in a matplotlib format.
plt.title("Food Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")
image = mpimg.imread("/Users/jacksusank/Downloads/P-ai/fridge_data/Food_Item_Detection.v1i.yolov7pytorch/train/images/001_png_jpg.rf.07d04531a296b96d3ffad3db54a54866.jpg")
# plt.imshow(image)
# plt.show()




