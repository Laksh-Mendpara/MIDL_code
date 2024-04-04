import numpy as np
import cv2
import numpy as np
import cv2
import random
import string
import os
import PIL
#import tensorflow as tf
import torch
import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
#from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
#from torch.utils.data import random_split
# %matplotlib inline
import os
import torch
import torchvision
import os
import random
from sklearn.model_selection import train_test_split
#import tarfile
#from torchvision.datasets.utils import download_url
#from torch.utils.data import random_split
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import log10, sqrt
import numpy as np
from PIL import Image, ImageChops
import math
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch, torchvision
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from einops import rearrange, repeat
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import math, os, copy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from PIL import Image
from SSIM_PIL import compare_ssim
import time


def add_artifacts(image, num_shapes):
    if image is None:
        print("Error: Unable to load the image.")
        return None
    image_with_artifacts = image.copy()
    height, width, channels = image_with_artifacts.shape

    for _ in range(num_shapes):
        # Randomly select a shape type
        #shape_type = random.choice(['circle', 'rectangle', 'polygon', 'triangle', 'alphabet'])
        shape_type = random.choice(['circle', 'rectangle'])
        if shape_type == 'circle':
            x = np.random.randint(5, width - 10)  # Avoid edges
            y = np.random.randint(5, height - 10)
            radius = np.random.randint(5, 10)  # Limit size
            color = (255, 255, 255)
            thickness = 1
            cv2.circle(image_with_artifacts, (x, y), radius, color, thickness)

        elif shape_type == 'rectangle':
            x1 = np.random.randint(5, width - 15)
            y1 = np.random.randint(5, height - 15)
            x2 = x1 + np.random.randint(5, 15)
            y2 = y1 + np.random.randint(5, 15)
            color = (255, 255, 255)
            thickness = 1
            cv2.rectangle(image_with_artifacts, (x1, y1), (x2, y2), color, thickness)

        elif shape_type == 'polygon':
            num_vertices = np.random.randint(1, 2)
            vertices = [(np.random.randint(10, width - 10), np.random.randint(10, height - 10)) for _ in range(num_vertices)]
            vertices = np.array(vertices, np.int32).reshape((-1, 1, 2))
            color = (255, 255, 255)
            thickness = 1
            cv2.polylines(image_with_artifacts, [vertices], isClosed=True, color=color, thickness=thickness)


        elif shape_type == 'alphabet':
            font = cv2.FONT_HERSHEY_SIMPLEX
            alphabet = random.choice(string.ascii_uppercase)
            font_scale = np.random.uniform(0.25, 0.75)
            font_thickness = 2
            color = (255, 255, 255)
            x = np.random.randint(10, width - 30)
            y = np.random.randint(10, height - 30)
            cv2.putText(image_with_artifacts, alphabet, (x, y), font, font_scale, color, font_thickness)

    return image_with_artifacts

from torch.utils.data import Dataset
from pathlib import Path
import pydicom
from torchvision.datasets import DatasetFolder
from PIL import Image
from torchvision.transforms.functional import to_tensor
def load_dicom_as_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    image = dicom_data.pixel_array
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize the image
    image = (image * 255).astype(np.uint8)  # Convert the image to uint8
    image_pil = Image.fromarray(image)
    return image_pil

def load_dicom_as_tensor(file_path):
    dicom_data = pydicom.dcmread(file_path)
    image = dicom_data.pixel_array
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize the image
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.unsqueeze(0)  
    image_tensor = image_tensor.unsqueeze(0)  
    image_tensor = image_tensor.expand(-1, 3, -1, -1)  
    image_tensor = image_tensor.squeeze()  
    transform = transforms.Resize((256, 256))
    image_tensor = transform(image_tensor)

    image_pil = transforms.ToPILImage()(image_tensor)
    return image_pil


import random

class DICOMDataset(Dataset):
    def __init__(self, root, loader, transform=None, num_artifacts=10, subset_percentage=1.00):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.num_artifacts = num_artifacts
        self.subset_percentage = subset_percentage  # Set the desired subset percentage
        self.img_files = [file for file in os.listdir(self.root) if file.endswith('.dicom')]
        
        # Randomly select a subset of the image files
        random.shuffle(self.img_files)
        subset_size = int(len(self.img_files) * self.subset_percentage)
        self.img_files = self.img_files[:subset_size]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.img_files[index])
        image_tensor = self.loader(file_path)

        # Add artifacts to the image
        if self.num_artifacts > 0:
            image_np = np.array(image_tensor)
            image_with_artifacts = add_artifacts(image_np, num_shapes=self.num_artifacts)
            image_tensor = Image.fromarray(image_with_artifacts)

        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, 0  # The second element is a dummy label

import os
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

# Define the parameters
batch_size = 1
LR_size = 64
img_size = 256

# Define the root directories for the training and test datasets
root = '/DATA2/VinDr-CXR/train'
testroot = '/DATA2/VinDr-CXR/test'
source_dir_train = Path(root)
source_dir_test = Path(testroot)

# Import necessary functions and classes

# Define transformations
transforms_ = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = DICOMDataset(root=root, loader=load_dicom_as_tensor, transform=transforms_, subset_percentage=1.00)
test_dataset = DICOMDataset(root=testroot, loader=load_dicom_as_tensor, transform=transforms_, subset_percentage=1.00)

# Split the training dataset into training and validation sets
train_data, val_data = train_test_split(train_dataset, test_size=0.2, random_state=42)

# Create dataloaders for training, validation, and test sets
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

# Define paths to save the dataloaders
save_path_train_dataloader = 'train_dataloader.pt'
save_path_val_dataloader = 'val_dataloader.pt'
save_path_test_dataloader = 'test_dataloader.pt'

# Save the dataloaders
torch.save(train_dataloader, save_path_train_dataloader)
torch.save(val_dataloader, save_path_val_dataloader)
torch.save(test_dataloader, save_path_test_dataloader)

print("DataLoaders saved successfully.")
