# Libraries for data handling and normalization
import os.path
import warnings
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Libraries used for data loading and prepaing the dataset subset
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import random
from torch.utils.data import random_split

# Library used for extracting features from the model using ResNet18
from torchvision.models import resnet18, ResNet18_Weights

# Library used for PCA dimensionality reduction
from sklearn.decomposition import PCA

# Set a random seed for reproducibility
random.seed(42)

# Directory to save the processed data
feature_dir = "./processed_data"
os.makedirs(feature_dir, exist_ok=True)

# Paths to save the processed data in a .pt file
train_features_path = os.path.join(feature_dir, "train_features_pca.pt")
test_features_path = os.path.join(feature_dir, "test_features_pca.pt")
train_labels_path = os.path.join(feature_dir, "train_labels.pt")
test_labels_path = os.path.join(feature_dir, "test_labels.pt")

# Function to save data to a file
def save_data(data, file_path):
    torch.save(data, file_path)
    print(f"Data being saved to {file_path}")

# Function to load data from a file
def load_data(file_path):
    warnings.simplefilter("ignore", category=FutureWarning)
    data = torch.load(file_path)
    print(f"Data is being loaded from {file_path}")
    return data

# Check if the processed data already exists and load it if it does
if all(os.path.exists(path) for path in [train_features_path, test_features_path, train_labels_path, test_labels_path]):
    print("Loading the already saved preprocessed data... \n")
    train_features_pca = load_data(train_features_path)
    test_features_pca = load_data(test_features_path)
    train_labels = load_data(train_labels_path)
    test_labels = load_data(test_labels_path)

# If preprocessed data does not exist, process the data
else:
    print("Processing data for the first time... \n")

    # Transform used for the CIFAR-10 dataset resize to 224x224 and normalize the data for ResNet18
    transform = transforms.Compose(
        [transforms.Resize((224,224)),     
        transforms.ToTensor(),             
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])      
    ])

    # Load the CIFAR-10 dataset and apply the transform
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Check size of the transformed CIFAR-10 dataset
    print(f"\nTrain Dataset size after the transform: {len(train_dataset)}")
    print(f"Test Dataset size after the transform: {len(test_dataset)}")
    print("Sample transformed image shape:", train_dataset[0][0].shape)

    # Limit set to the number of images in the dataset to 500 training and 100 testing images per class
    train_images_per_class = 500
    test_images_per_class = 100

    # Number of classes needed for CIFAR-10
    num_classes = 10

    # Function to get a subset of the dataset without any overlapping classes
    def get_class_subset(dataset, class_count, exclude_indices=None):
        # Initialize a dictionary to keep track of the class counts
        class_counts = {i: 0 for i in range(num_classes)}

        # Shuffle the indices to avoid any overlapping between classes
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        exclude_indices = exclude_indices or set()

        # Get a subset of the dataset without any overlapping classes
        subset_indices = []
        # Iterate over the indices to add images to the subset 
        for idx in indices:
            _, label = dataset[idx]
            # Check if the class count for the current label is less than the class count and the index is not in the exclude indices
            if idx not in exclude_indices and class_counts[label] < class_count:
                subset_indices.append(idx)
                class_counts[label] += 1
            # Check if all the class counts are bigger than the class count
            if all(count >= class_count for count in class_counts):
                break

        return Subset(dataset, subset_indices), set(subset_indices)

    # Get the subset of the dataset
    train_data, train_indices = get_class_subset(train_dataset, train_images_per_class)
    test_data, test_indices = get_class_subset(test_dataset, test_images_per_class, exclude_indices=train_indices)

    # Check size of the subset dataset
    print(f"\nSubset train data size: {len(train_data)}")
    print(f"Subset test data size: {len(test_data)}")
    print("Sample transformed image shape:", train_data[0][0].shape)

    # Data loaders for batching and shuffling
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Device used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load the pre-trained ResNet18 model with the default weights 
    resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Remove the last layer of the ResNet18 model to get the features
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])

    # Move model to cuda or CPU device
    resnet_model = resnet_model.to(device)

    # Set the model to evaluation mode to get the features
    resnet_model.eval()

    # Function to extract the features and labels from the model
    def extract_features(model, loader):
        # Initialize lists to store the features and labels
        features = []
        labels = []
        # Loop over the dataset to get the features and labels
        with torch.no_grad():
            for images, target in loader:
                # Pass image through the model to get the features
                output = model(images)
                output = output.view(output.size(0), -1) # Flatten features
                # Add the features and labels to the lists
                features.append(output)
                labels.append(target)

        # Concatenate the features and labels
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels

    # Extract the features from the model (512 x 1)
    train_features, train_labels = extract_features(resnet_model, train_loader)
    test_features, test_labels = extract_features(resnet_model, test_loader)

    # Check size of the features and labels 
    print(f"\nTrain features: {train_features.shape}")
    print(f"Train labels: {train_labels.shape}")
    print(f"Test features: {test_features.shape}")
    print(f"Test labels: {test_labels.shape}")

    # Initialize PCA, but further reduce the size of feature bector from 512 x 1 to 50 x 1
    pca = PCA(n_components=50)

    # Fit the PCA model on the training features and transform the features
    train_features_pca = pca.fit_transform(train_features)
    test_features_pca = pca.transform(test_features)

    # Check size of the PCA transformation
    print(f"\nTrain features after PCA: {train_features_pca.shape}") 
    print(f"Test features after PCA: {test_features_pca.shape}")

    # Save processed data
    save_data(train_features_pca, train_features_path)
    save_data(test_features_pca, test_features_path)
    save_data(train_labels, train_labels_path)
    save_data(test_labels, test_labels_path)
    print("Processed data saved.")

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
