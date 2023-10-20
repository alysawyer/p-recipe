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
# T.ToPILImage()(white_torch)
T.ToPILImage()(white_torch).show()

