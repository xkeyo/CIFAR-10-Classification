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

# Import the evaluation functions to show the performance of the models
from evaluation import evaluate_model, summarize_metrics

# Function to save the model to a file
def save_model(model, file_path):
    torch.save(model, file_path)
    print(f"Model saved to {file_path}")

# Function to load the model from a file
def load_model(file_path):
    warnings.simplefilter("ignore", category=FutureWarning)
    model = torch.load(file_path)
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
        "sklearn_naives_bayes": os.path.join(model_dir, "sklearn_naives_bayes.pt")
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





    # Summarize the metrics for all the tested models
    summarize_metrics(metrics_summary)


