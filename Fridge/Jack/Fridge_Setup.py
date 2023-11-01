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
from PIL import Image
import splitfolders

import locale

# Set the locale to en_US
locale.setlocale(locale.LC_ALL, 'en_US')


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# print(torch. __version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


use_cuda = torch.cuda.is_available()
# print(use_cuda)



""" These lines just open the picture of apple pie in the first dataset.
white_torch = torchvision.io.read_image("/Users/jacksusank/Downloads/p-ai/fridge_data/food_data_set_training/images/apple_pie/134.jpg")
print("This is a picture of apple pie")
T.ToPILImage()(white_torch).show()
"""









# Next, I'm going to try to make and open an image that is just the box which is labeled 

# old  label_file_path = "/Users/jacksusank/Downloads/P-ai/fridge_data/Food_Item_Detection.v1i.yolov7pytorch/train/labels/001_png_jpg.rf.07d04531a296b96d3ffad3db54a54866.txt"  # Replace with the path to your YOLO label file
new_label_file_path = "/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train/labels/asparagus_jpg.rf.8b91337853f5dd660fff7bf6c9e058b9.txt"
new_image_file_path = "/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train/images/asparagus_jpg.rf.8b91337853f5dd660fff7bf6c9e058b9.jpg"

# This is a list of all the possible types of labels.
# old   names = ["Apple", 'Banana', 'Beans', 'Capsicum', 'Carrot', 'Cucumber', 'Curli-Flower', 'Orange', 'Tomato', 'Tomatos', 'apple', 'asparagus', 'avocado', 'banana', 'beef', 'bell_pepper', 'bento', 'blueberries', 'bottle', 'bread', 'broccoli', 'butter', 'can', 'carrot', 'cauliflower', 'cheese', 'chicken', 'chicken_breast', 'chocolate', 'coffee', 'corn', 'cucumber', 'egg', 'eggs', 'energy_drink', 'fish', 'flour', 'garlic', 'goat_cheese', 'grapes', 'grated_cheese', 'green_beans', 'ground_beef', 'guacamole', 'ham', 'heavy_cream', 'humus', 'juice', 'ketchup', 'kothmari', 'leek', 'lemon', 'lettuce', 'lime', 'mango', 'marmelade', 'mayonaise', 'milk', 'mushrooms', 'mustard', 'nuts', 'onion', 'orange', 'pak_choi', 'parsley', 'peach', 'pear', 'pineapple', 'plasticsaveholder', 'pot', 'potato', 'potatoes', 'pudding', 'red_cabbage', 'red_grapes', 'rice_ball', 'salad', 'sandwich', 'sausage', 'shrimp', 'smoothie', 'spinach', 'spring_onion', 'strawberries', 'sugar', 'sweet_potato', 'tea_a', 'tea_i', 'tomato', 'tomato_sauce', 'tortillas', 'turkey', 'watermelon', 'yogurt']
names = ["Lemon", 'Onion', 'Orange', 'Peas', 'Potato', 'Strawberry', 'Tomato', 'apples', 'aubergine', 'bananas', 'blueberries', 'bread', 'broccoli', 'butter', 'carrots', 'cheese', 'chicken', 'courgettes', 'eggs', 'ginger', 'green beans', 'green chilies', 'ham', 'lemon', 'lettuce', 'lime', 'milk', 'mushrooms', 'onion', 'orange', 'peach', 'peppers', 'potatoes', 'red onion', 'spinach', 'spring onion', 'strawberries', 'sweet_potato', 'tomatoes', 'yoghurt']


def show_labels(label_file_path, image_file_path, category_values_list):
    """ This function takes the locations of an image and its corresponding label file and it shows cropped versions of the image that only show the labels.
    label_file_path: (String) This paramater is the pathname of the label file that corresponds to the image.
    image_file_path: (String) This parameter is the pathname of the image file that we will be cropping.
    category_values_list: (List of strings) This parameter is a list which contains all of the possible names of the labels in the same order that they were encoded by the dataset's creator.
    """

    im = Image.open(new_image_file_path) # I had to comment out the "r" for read file here because it was giving me errors.
    width, height = im.size

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
        




        # Process the extracted information 
        print(f"Class ID: {class_id}, Center X: {center_x}, Center Y: {center_y}, Width: {label_width}, Height: {label_height}")


        # I'm not sure why the bottom_bound is bigger than the top_bound, but I think it might be because it's being measured from the top? I might need to rename my variables.
        # In these lines I am multiplying the Yolo float values which are between 0 and 1 by the width and height of the original picture so that I get the locations of the pixels.
        print("This is a " + str(category_values_list[class_id]))
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
        im1.save("/Users/jacksusank/Downloads/p-ai/fridge_data/Our_Data/" + str(category_values_list[class_id]) + "/image1.jpg")

show_labels(new_label_file_path, new_image_file_path, names)


#Cheese plate folder path
my_folder_path = "/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train"


# Define a function you want to apply to each file
images_pathname_list = []
labels_pathname_list = []

def make_cropped_pictures_folder(folder_path):
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if subfolder_path == "/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train/images":
            num = 0
            for filename in os.listdir(folder_path):
                num += 1
                # Construct the full file path
                images_file_path = os.path.join(folder_path, filename)
                images_pathname_list.append(images_file_path)
        elif subfolder_path == "/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train/labels":
            num = 0
            for filename in os.listdir(folder_path):
                num += 1
                # Construct the full file path
                labels_file_path = os.path.join(folder_path, filename)
                labels_pathname_list.append(labels_file_path)
    for i in range(len(10)):
        try:
            with Image.open(images_pathname_list[i]) as img:
                # Image was successfully opened, you can work with it here
                show_labels(images_pathname_list[i], labels_pathname_list[i], names)
        except:
            # Handle the case where the file is not a valid image

            print(f"The file at '{images_pathname_list[i]}' is not a valid image file.")












# Testing the function on an example in the SmarterChef training data
# show_labels("/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train/labels/asparagus_jpg.rf.8b91337853f5dd660fff7bf6c9e058b9.txt", "/Users/jacksusank/Downloads/p-ai/fridge_data/SmarterChef.v5i.yolov7pytorch/train/images/asparagus_jpg.rf.8b91337853f5dd660fff7bf6c9e058b9.jpg", names)









# This block of code will display the original image that goes along with these labels in a matplotlib format.
plt.title("Food Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")
image = mpimg.imread("/Users/jacksusank/Downloads/P-ai/fridge_data/Food_Item_Detection.v1i.yolov7pytorch/train/images/001_png_jpg.rf.07d04531a296b96d3ffad3db54a54866.jpg")
# plt.imshow(image)
# plt.show()






# Working with the model! Resnet 50
model = models.resnet50(pretrained=True)
model
# print(model)

num_features = model.fc.in_features 
print('Number of features from pre-trained model', num_features)


model.fc = nn.Linear(num_features, 101)
model = model.to(device)

requires_grad=False



# Setting up the datasets
# train_dir = "/Users/jacksusank/Downloads/p-ai/fridge_data/Food_Item_Detection.v1i.yolov7pytorch/train/images"
# test_dir = "/Users/jacksusank/Downloads/p-ai/fridge_data/Food_Item_Detection.v1i.yolov7pytorch/test/images" 
train_dir = "/Users/jacksusank/Downloads/p-ai/fridge_data/food_data_set_training/processed_data/train"
test_dir = "/Users/jacksusank/Downloads/p-ai/fridge_data/food_data_set_training/processed_data/test"

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transforms_train)
test_dataset = datasets.ImageFolder(test_dir, transforms_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)






# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)



#### Train model
train_loss=[]
train_accuary=[]
test_loss=[]
test_accuary=[]


num_epochs = 30   #(set no of epochs)
start_time = time.time() #(for showing time)

# Start loop
for epoch in range(num_epochs): #(loop for every epoch)
    print("Epoch {} running".format(epoch)) #(printing message)

    """ Training Phase """

    model.train()    #(training model)
    running_loss = 0.   #(set loss 0)
    running_corrects = 0 
    # load a batch data of images
    for i, (inputs, labels) in enumerate(train_dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device) 
        # forward inputs and get output
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # get loss value and update the network weights
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()
        
        # print("Finished input " + str(i))
        print(len(train_dataloader))
    print("Done with step 2")
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset) * 100.
    # Append result
    train_loss.append(epoch_loss)
    train_accuary.append(epoch_acc)
    # Print progress
    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time() -start_time))

    """Testing Phase"""

    model.eval()
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        # Append result
        test_loss.append(epoch_loss)
        test_accuary.append(epoch_acc)
        # Print progress
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch+1, epoch_loss, epoch_acc, time.time()- start_time))

