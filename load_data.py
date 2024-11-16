# Libraries for data handling and processing
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Import CIFAR-10 dataset
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

# Import ResNet18 model
from torchvision.models import resnet18, ResNet18_Weights

# Libraries for data visualization
import matplotlib.pyplot as plt
import numpy as np

# Library for PCA
from sklearn.decomposition import PCA

# Transform used for the CIFAR-10 dataset resize to 224x224 and normalize the data for ResNet18
transform = transforms.Compose(
    [transforms.Resize((224,224)),     
     transforms.ToTensor(),             
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])      
])

# Load the CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Filter used to limit the number of images in the dataset for 500 training and 100 testing images per class
train_images_per_class = 500
test_images_per_class = 100

# Number of classes in CIFAR-10
num_classes = 10

# Function to get a subset of the dataset with a specified number of images per class
def get_class_subset(dataset, class_count):
    class_counts = {i: 0 for i in range(num_classes)}
    indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < class_count:
            indices.append(idx)
            class_counts[label] += 1
        # Stop when we have enough samples for each class
        if all(count >= class_count for count in class_counts.values()):
            break

    return Subset(dataset, indices)

# Filter dataset for 500 training and 100 testing images per class
train_data = get_class_subset(train_dataset, train_images_per_class)
test_data = get_class_subset(test_dataset, test_images_per_class)

# Data loaders for batching
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load the pre-trained ResNet18 model with the default weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last layer

# Set the model to evaluation mode
model.eval()

# Function to extract the features and labels from the model 
def extract_features(model, loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, target in loader:
            # Pass image through the model to get the features
            output = model(images)
            output = output.view(output.size(0), -1) # Flatten features
            features.append(output)
            labels.append(target)

    # Concatenate the features and labels
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels

# Extract the features from the model (512 x 1)
train_features, train_labels = extract_features(model, train_loader)
test_features, test_labels = extract_features(model, test_loader)

# Initialize PCA, but further reduce the size of feature bector from 512 x 1 to 50 x 1
pca = PCA(n_components=50)

# Fit the PCA model on the training features and transform the features
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)

# Getters for the transformed features
def get_train_features_pca():
    return train_features_pca

def get_test_features_pca():    
    return test_features_pca

# Getters for labels
def get_train_labels():
    return train_labels

def get_test_labels():    
    return test_labels

# # Print the shape of the transformed features
# print(train_features_pca.shape)
# print(test_features_pca.shape)

# # Display the amount of variance explained by the 50 components
# print("Explained variance by 50 PCA components:", sum(pca.explained_variance_ratio_))