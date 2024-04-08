import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from pathlib import Path
import pydicom
from torchvision.datasets import DatasetFolder
from PIL import Image
from torchvision.transforms.functional import to_tensor

def add_artifacts(image, num_shapes):
    if image is None:
        print("Error: Unable to load the image.")
        return None
    image_with_artifacts = image.copy()
    height, width, channels = image_with_artifacts.shape

    for _ in range(num_shapes):
        # Randomly select a shape type
        shape_type = random.choice(['circle', 'rectangle', 'polygon', 'triangle', 'alphabet'])

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
    def __init__(self, root, loader, csv_file, transform=None, num_artifacts=25, subset_percentage=0.5):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.num_artifacts = num_artifacts
        self.subset_percentage = subset_percentage

        # Load the CSV file containing bounding box info
        try:
            self.csv_data = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

        if 'image_id' not in self.csv_data.columns:
            raise ValueError("CSV file must contain 'image_id' column.")

        # List of image files in the root directory with the ".dicom" extension
        self.img_files = [file for file in os.listdir(self.root) if file.endswith('.dicom')]
        random.shuffle(self.img_files)

        # Calculate the size of the subset based on the specified percentage
        subset_size = int(len(self.img_files) * self.subset_percentage)
        # Select a random subset of image files
        self.img_files = self.img_files[:subset_size]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Get the file path of the current image
        file_name = self.img_files[index]
        file_path = os.path.join(self.root, file_name)

        # Load the image using the provided loader function
        image_tensor = self.loader(file_path)

        # Find bounding box data for the current image index
        bbox_data = self.csv_data[self.csv_data.index == index]

        if bbox_data.empty:
            warnings.warn(f"No bounding box data found for image index: {index}")
            bounding_boxes = torch.zeros((0, 4), dtype=torch.float32)  # Empty bounding box tensor
        else:
            # Process bounding box data as needed (e.g., extract coordinates)
            bounding_boxes = bbox_data[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(float)

        # Add artifacts to the image if specified
        if self.num_artifacts > 0:
            image_np = np.array(image_tensor)
            image_with_artifacts = add_artifacts(image_np, num_shapes=self.num_artifacts)
            image_tensor = Image.fromarray(image_with_artifacts)

        # Apply the specified transformation to the image if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, bounding_boxes, file_name, 0
        #return image_tensor, 0


import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


# Define the paths and other necessary variables
batch_size = 1
img_size = 256
root = '/DATA2/VinDr-CXR/train'
testroot = '/DATA2/VinDr-CXR/test'

# Load CSV files
c1 = '/DATA2/VinDr-CXR/annotations/annotations_test.csv'
c2 = '/DATA2/VinDr-CXR/annotations/annotations_train.csv'
c3 = '/DATA2/VinDr-CXR/annotations/image_labels_test.csv'
c4 = '/DATA2/VinDr-CXR/annotations/image_labels_train.csv'

cc1 = pd.read_csv(c1)
cc3 = pd.read_csv(c3)
test_csv = pd.merge(cc1, cc3, on='image_id', how='inner')
test_csv.to_csv('/home/dattatreyo/sr3_try/test_csv.csv', index=False)

cc2 = pd.read_csv(c2)
cc4 = pd.read_csv(c4)
train_csv = pd.merge(cc1, cc3, on='image_id', how='inner')

train_csv.to_csv('/home/dattatreyo/sr3_try/train_csv.csv', index=False)

columns_to_replace = ['x_min', 'y_min', 'x_max', 'y_max']
train_csv[columns_to_replace] = train_csv[columns_to_replace].fillna(0)
test_csv[columns_to_replace] = test_csv[columns_to_replace].fillna(0)

train_csv.to_csv('modified_train.csv', index=False)
test_csv.to_csv('modified_test.csv', index=False)

# Load the DICOMDataset class and necessary functions here...
# Define the transformations for the datasets
transforms_ = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transforms2 = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

# Create train and validation datasets
train_dataset = DICOMDataset(root=root, loader=load_dicom_as_tensor, csv_file='/home/dattatreyo/sr3_try/modified_train.csv', transform=transforms_, subset_percentage=0.025)
val_dataset = DICOMDataset(root=root, loader=load_dicom_as_tensor, csv_file='/home/dattatreyo/sr3_try/modified_train.csv', transform=transforms_, subset_percentage=0.025)
test_dataset = DICOMDataset(root=testroot, loader=load_dicom_as_tensor, csv_file='/home/dattatreyo/sr3_try/modified_test.csv', transform=transforms2, subset_percentage=0.025)

# Split the dataset into train and validation sets
train_files, val_files = train_test_split(train_dataset.img_files, test_size=0.2, random_state=42)

# Create DataLoaders for training, validation, and test
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

# Save the dataloaders
torch.save(train_loader, 'train_loader.pth')
torch.save(val_loader, 'val_loader.pth')
torch.save(test_loader, 'test_loader.pth')
