# Library used for file path, saving and loading models, and managing warnings
import os.path
import torch
import warnings

# Library used for data visualization and manipulation
import numpy as np
import matplotlib.pyplot as plt

# Import the data loading functions from the load_data file
from load_data import get_test_features_pca, get_train_features_pca, get_train_labels, get_test_labels

# Import the Gaussian Naive Bayes model frpom the naives_bayes file and the sklearn Naive Bayes model
from naives_bayes import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB

# Import the Decision Tree model from the tree file and the sklearn Decision Tree model
from sklearn.tree import DecisionTreeClassifier
from tree import DecisionTree

# Import the MLP models from the mlp file
from MLP.mlp_functions import train_mlp_model, predict_mlp
import MLP.mlp as mlp_base
import MLP.mlp_variant1 as mlp_1
import MLP.mlp_variant2 as mlp_2
import MLP.mlp_variant3 as mlp_3
import MLP.mlp_variant4 as mlp_4
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from cnn import VGG11, train_model, predict
from cnn2 import VGG11LargerKernel

from torchvision.datasets import CIFAR10
# Import the evaluation functions to show the performance of the models
from evaluation import evaluate_model, summarize_metrics


# Function to save the model to a file
def save_model(model, file_path):
    torch.save(model, file_path)
    print(f"Model saved to {file_path}")

# Function to load the model from a file
def load_model(file_path):
    warnings.simplefilter("ignore", category=FutureWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(file_path, map_location=device)
    print(f"Model loaded from {file_path}")
    return model

def main():
    return 0

if __name__ == "__main__":

    # Directory to save the models
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    # Dictionary used to store the different file path used to save and load the model
    file_path = {
        "custom_naives_bayes": os.path.join(model_dir, "custom_naives_bayes.pt"),
        "sklearn_naives_bayes": os.path.join(model_dir, "sklearn_naives_bayes.pt"),
        "sklearn_decision_tree": os.path.join(model_dir, "sklearn_decision_tree.pt"),
        "custom_decision_tree": os.path.join(model_dir, "custom_decision_tree.pt"),
        "custom2_decision_tree": os.path.join(model_dir, "custom2_decision_tree.pt"),
        "mlp_base": os.path.join(model_dir, "mlp_base.pt"),
        "mlp_1": os.path.join(model_dir, "mlp_1.pt"),
        "mlp_2": os.path.join(model_dir, "mlp_2.pt"),
        "mlp_3": os.path.join(model_dir, "mlp_3.pt"),
        "mlp_4": os.path.join(model_dir, "mlp_4.pt"),
        "cnn": os.path.join(model_dir, "cnn.pt"),
        "cnn2": os.path.join(model_dir, "cnn2.pt")
    }

    # Get the training features and labels and tesing features and labels
    train_features_pca = get_train_features_pca()
    train_labels = get_train_labels()
    test_features_pca = get_test_features_pca()
    test_labels = get_test_labels()
    class_labels = list(range(10)) # CIFAR-10 has 10 classes (0-9)

    # Dictionary used to store the different metrics for each model
    metrics_summary = {}

    # -------------------- Customly implemented Gaussian Naive Bayes model -------------------- #
    
    if os.path.exists(file_path["custom_naives_bayes"]):
        # Load model from file
        print(f"\nLoading model from {file_path['custom_naives_bayes']}")
        custom_naives_bayes = load_model(file_path["custom_naives_bayes"])
    else:
        # Train the model
        print("\nTraining custom Gaussian Naive Bayes model...")
        custom_naives_bayes = GaussianNaiveBayes()
        custom_naives_bayes.fit(train_features_pca, train_labels)

        # Save the model to file
        save_model(custom_naives_bayes, file_path["custom_naives_bayes"])

    # Evaluate the Custom Gaussian Naive Bayes model
    print("\nEvaluating custom Gaussian Naive Bayes model...")
    custom_naives_bayes_predictions = custom_naives_bayes.predict(test_features_pca)

    # Get the metrics for the Custom Gaussian Naive Bayes model
    custom_naives_bayes_metrics = evaluate_model("Custom Gaussian Naive Bayes", custom_naives_bayes_predictions, test_labels.numpy(), class_labels)
    # Save the metrics for the Custom Gaussian Naive Bayes model
    metrics_summary["Custom Naive Bayes"] = custom_naives_bayes_metrics
    print(f"\nCustom Gaussian Naive Bayes model metrics: {custom_naives_bayes_metrics}")

    # -------------------- Sklearn implemented Gaussian Naive Bayes model -------------------- #

    if os.path.exists(file_path["sklearn_naives_bayes"]):
        # Load model from file
        print(f"\nLoading model from {file_path['sklearn_naives_bayes']}")
        sklearn_naives_bayes = load_model(file_path["sklearn_naives_bayes"])
    else:
        # Train the model
        print("\nTraining sklearn Gaussian Naive Bayes model...")
        sklearn_naives_bayes = GaussianNB()
        sklearn_naives_bayes.fit(train_features_pca, train_labels)

        # Save the model to file
        save_model(sklearn_naives_bayes, file_path["sklearn_naives_bayes"])

    # Evaluate the Sklearn Gaussian Naive Bayes model
    print("\nEvaluating sklearn Gaussian Naive Bayes model...")
    sklearn_naives_bayes_predictions = sklearn_naives_bayes.predict(test_features_pca)

    # Get the metrics for the Sklearn Gaussian Naive Bayes model
    sklearn_naives_bayes_metrics = evaluate_model("Sklearn Gaussian Naive Bayes", sklearn_naives_bayes_predictions, test_labels.numpy(), class_labels)
    # Save the metrics for the Sklearn Gaussian Naive Bayes model
    metrics_summary["Sklearn Naive Bayes"] = sklearn_naives_bayes_metrics
    print(f"\nSklearn Gaussian Naive Bayes model metrics: {sklearn_naives_bayes_metrics}")

  # -------------------- Sklearn implemented Decision Tree model -------------------- #

    if os.path.exists(file_path["sklearn_decision_tree"]):
        # Load model from file
        print(f"\nLoading model from {file_path['sklearn_decision_tree']}")
        sklearn_decision_tree = load_model(file_path["sklearn_decision_tree"])
    else:
        # Train the model
        print("\nTraining sklearn Decision Tree model...")
        sklearn_decision_tree = DecisionTreeClassifier(max_depth=50)
        sklearn_decision_tree.fit(train_features_pca, train_labels)

        # Save the model to file
        save_model(sklearn_decision_tree, file_path["sklearn_decision_tree"])

    # Evaluate the Sklearn Decision Tree model
    print("\nEvaluating sklearn Decision Tree model...")
    sklearn_decision_tree_predictions = sklearn_decision_tree.predict(test_features_pca)

    # Get the metrics for the Sklearn Decision Tree model
    sklearn_decision_tree_metrics = evaluate_model("Sklearn Decision Tree", sklearn_decision_tree_predictions, test_labels.numpy(), class_labels)
    # Save the metrics for the Sklearn Decision Tree model
    metrics_summary["Sklearn Decision Tree"] = sklearn_decision_tree_metrics
    print(f"\nSklearn Decision Tree model metrics: {sklearn_decision_tree_metrics}")

 # -------------------- Customly implemented Decision Tree model -------------------- #

    if os.path.exists(file_path["custom_decision_tree"]):
        # Load model from file
        print(f"\nLoading model from {file_path['custom_decision_tree']}")
        custom_decision_tree = load_model(file_path["custom_decision_tree"])
    else:
        # Train the model
        print("\nTraining custom Decision Tree model...")
        custom_decision_tree = DecisionTree(max_depth=50)
        custom_decision_tree.fit(train_features_pca, train_labels)

        # Save the model to file
        save_model(custom_decision_tree, file_path["custom_decision_tree"])

    # Evaluate the Custom Decision Tree model
    print("\nEvaluating custom Decision Tree model...")
    custom_decision_tree_predictions = custom_decision_tree.predict(test_features_pca)

    # Get the metrics for the Custom Decision Tree model
    custom_decision_tree_metrics = evaluate_model("Custom Decision Tree", custom_decision_tree_predictions, test_labels.numpy(), class_labels)
    # Save the metrics for the Custom Decision Tree model
    metrics_summary["Custom Decision Tree"] = custom_decision_tree_metrics
    print(f"\nCustom Decision Tree model metrics: {custom_decision_tree_metrics}")
    
    
     # -------------------- Customly implemented Decision Tree model (with larger depth) -------------------- #

    if os.path.exists(file_path["custom2_decision_tree"]):
        # Load model from file
        print(f"\nLoading model from {file_path['custom2_decision_tree']}")
        custom_decision_tree = load_model(file_path["custom2_decision_tree"])
    else:
        # Train the model
        print("\nTraining custom Decision Tree model...")
        custom_decision_tree = DecisionTree(max_depth=75)
        custom_decision_tree.fit(train_features_pca, train_labels)

        # Save the model to file
        save_model(custom_decision_tree, file_path["custom2_decision_tree"])

    # Evaluate the Custom Decision Tree model
    print("\nEvaluating custom Decision Tree model...")
    custom_decision_tree_predictions = custom_decision_tree.predict(test_features_pca)

    # Get the metrics for the Custom Decision Tree model
    custom_decision_tree_metrics = evaluate_model("Custom Decision Tree", custom_decision_tree_predictions, test_labels.numpy(), class_labels)
    # Save the metrics for the Custom Decision Tree model
    metrics_summary["Custom Decision Tree (deeper)"] = custom_decision_tree_metrics
    print(f"\nCustom Decision Tree model metrics: {custom_decision_tree_metrics}")

    # -------------------- Multi-layer Perceptron model (MLP)-------------------- #

    # Dictionary of all the different MLP models
    models = {
        "mlp_base": mlp_base.MLP(),
        "mlp_1": mlp_1.MLP(),
        "mlp_2": mlp_2.MLP(),
        "mlp_3": mlp_3.MLP(),
        "mlp_4": mlp_4.MLP()
    }

    # Convert featutures and lables to tensors
    train_features_tensor = torch.tensor(train_features_pca, dtype=torch.float32)
    train_labels_tensor = train_labels.clone().detach().to(dtype=torch.long)
    test_features_tensor = torch.tensor(test_features_pca, dtype=torch.float32)
    test_labels_tensor = test_labels.clone().detach().to(dtype=torch.long)

    # Loop over the MLP models
    for model_name, model in models.items():

        # Check if the model is already trained then load it
        if os.path.exists(file_path[model_name]):
            # Load model from file
            print(f"\nLoading model from {file_path[model_name]}")
            model = load_model(file_path[model_name])
        
        # If model is not trained, train it
        else:
            # Train the model
            print(f"\nTraining {model_name} model...")

            # Train the model
            train_mlp_model(model, train_features_tensor, train_labels_tensor, num_epochs=20, batch_size=32, learning_rate=0.001)

            # Save the model to file
            save_model(model, file_path[model_name])

        # Evaluate the MLP model
        print(f"\nEvaluating {model_name} model...")

        # Convert featutures and lables to tensors
        predictions = predict_mlp(model, test_features_tensor)

        # Get the metrics for the MLP model
        metrics = evaluate_model(model_name, predictions, test_labels_tensor.numpy(), class_labels)
        # Save the metrics for the MLP model
        metrics_summary[model_name] = metrics
        print(f"\n{model_name} model metrics: {metrics}")



# -------------------- Convolutional Neural Network models (CNN)-------------------- #
# Print the number of test labels
    print(torch.cuda.is_available())  
    #print(torch.cuda.get_device_name(0))  
    
  # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}")
    
    # Define data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # transform = transforms.Compose(
    #     [transforms.Resize((224,224)),     
    #     transforms.ToTensor(),             
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225])      
    # ])
    
    # Load the CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
# -------------------- Convolutional Neural Network models 1 (CNN)-------------------- #
    # CNN Training and Evaluation
    if os.path.exists(file_path["cnn"]):
        print(f"\nLoading cnn model from {file_path['cnn']}")
        cnn = load_model(file_path["cnn"])
    else:


        # Now fit the model
        print("\nTraining cnn model...")
        cnn = VGG11(num_classes=10).to(device)  # Ensure the device is CUDA or CPU
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
        num_epochs = 10
        
        train_model(cnn, train_loader, criterion, optimizer, num_epochs, device)
        save_model(cnn, file_path["cnn"])

    # Evaluate the CNN model
    print("\nEvaluating CNN model...")
    predictions = predict(cnn.to(device), test_loader, device)

    cnn_metrics = evaluate_model("cnn", predictions[:len(test_labels)], test_labels.numpy(), class_labels)
    
    
    # Save the metrics for the Custom Decision Tree model
    metrics_summary["cnn"] = cnn_metrics
    print(f"\nCNN model metrics: {cnn_metrics}")
    
# -------------------- Convolutional Neural Network models - Larger Kernel (CNN)-------------------- #

        # CNN Training and Evaluation
    if os.path.exists(file_path["cnn2"]):
        print(f"\nLoading cnn2 model from {file_path['cnn2']}")
        cnn = load_model(file_path["cnn2"])
    else:


        # Now fit the model
        print("\nTraining cnn2 model...")
        cnn = VGG11LargerKernel(num_classes=10).to(device)  # Ensure the device is CUDA or CPU
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
        num_epochs = 10
        
        train_model(cnn, train_loader, criterion, optimizer, num_epochs, device)
        save_model(cnn, file_path["cnn2"])

    # Evaluate the CNN model
    print("\nEvaluating CNN2 model...")
    predictions = predict(cnn.to(device), test_loader, device)

    cnn_metrics = evaluate_model("cnn2", predictions[:len(test_labels)] , test_labels.numpy(), class_labels)
    # Save the metrics for the Custom Decision Tree model
    metrics_summary["cnn2"] = cnn_metrics
    print(f"\nCNN2 model metrics: {cnn_metrics}")
    # -------------------- Summary -------------------- #

    # Summarize the metrics for all the tested models
    summarize_metrics(metrics_summary)





