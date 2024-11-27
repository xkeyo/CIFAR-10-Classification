# COMP-472: Artificial Intelligence Project FALL-2024 
## CIFAR-10 Classification Project
This project implements and evaluates various models for classification tasks on the CIFAR-10 image data. 
The models includes custom implementations of Gaussian Naive Bayes and Decision Trees
### **Note** 
- the cnn.pt file from training the cnn and the data directory were too  large to be commited to git. You can find them in a google drive folder at this link: https://drive.google.com/drive/folders/1LtBqQXRZ44SB2jDJ9PGLq7inKrVjme-Z?usp=drive_link 

## Files

- **`load_data.py`**: Handles data loading, preprocessing, feature extraction using ResNet18, and PCA for dimensionality reduction.
- **`naives_bayes.py`**: Custom implementation of Gaussian Naive Bayes model including the training and testing functions
- **`tree.py`**: Implementation of a custom Decision Tree model including the training and testing functions
- **`/MLP`**: Contains all the different MLP models including the base MLP model.
  - **`mlp_functions.py`**: Contains the training and testing function used for all the MLP models in order to train and test the models
  - **`mlp.py`**: Base implementation of a Multi-Layer Perceptron (MLP).
  - **`mlp_variant1.py`**: Implmentation of a Multi-Layer Perceptron (MLP) with a reduced depth of the network layers.
  - **`mlp_variant2.py`**: Implmentation of a Multi-Layer Perceptron (MLP) with a extended depth of the network layer
  - **`mlp_variant3.py`**: Implmentation of a Multi-Layer Perceptron (MLP) with a smaller size of the hidden layers.
  - **`mlp_variant4.py`**: Implmentation of a Multi-Layer Perceptron (MLP) with a bigger size of the hidden layers
- **`cnn.py`**: Base implementation of a Convolutional Neural Network (CNN).
- **`evaluation.py`**: Contains utility functions to evaluate models, including metrics computation and confusion matrix visualization.
- **`main.py`**: Main script for training, evaluating, and comparing models, also contains the sklearn model implementation for the Naive Bayes and Decision Tree.
- **`/processed_data`**: Contains all the data that has already been pre-processed (Train Features, Train Labels, Test Features, and Test Labels)
- **`/models`**: Contains all the different models used in the Project that have already been trained on the data (Custom Naive Bayes, SkLearn Naive Bayes, Custom Decision Tree, SkLearn Decision Tree, Base MLP model, 4 variants of the MLP models, and CNN model

## Setup

### Prerequisites
- Python 3.7+
- Libraries: `torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`

## Installation
1. Clone the repository:
   ```bash
    git clone <repository-url>
    cd <repository-folder>
   ```
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Alternatively, for conda users, install the dependencies using:
   ```bash
      conda env create -f environment.yml
   ```

## Steps to Execute the Code

### **1. Data Preprocessing**
The data preprocessing is managed in `load_data.py`. It performs the following tasks:
- Downloads and transforms the CIFAR-10 dataset.
- Extracts features using ResNet18.
- Applies PCA for dimensionality reduction.
- Saves the preprocessed data to `/processed_data`.

#### **Important Notes:**
- If the preprocessed files already exist in `/processed_data`, the script will load them directly without reprocessing the data.
- **Do not delete the preprocessed files** unless absolutely necessary. Deleting these files will trigger the preprocessing to run again, which will change the test data values and lead to results that differ from the metrics reported in the Report.

#### **IF AND ONLY IF YOU WANT TO REPROCESS THE DATA THEN:**
1. Delete the files in `/processed_data`.
2. Run the preprocessing script again:
   ```bash
   python load_data.py
   ```
Be aware that the testing dataset and results will differ from previously values 

### **2. Training, Evaluating, and Applying the Models**
The **`main.py`** script handles all of this:
- Loading the preprocessed data
- Training the models
- Evaluating the models
- Displaying the metrics
- Saving the trained models

To Execute the Training and Evaluation:
Simply run the **`main.py`** script
  ```bash
      python main.py
  ```
- The script checks if a trained model already exists in /models:
  - If the model exists, it loads the saved model and skips training and evaluates it.
  - If the model does not exist, it trains the model, evaluates it, and saves it to /models.
- No need to reprocess data or manually train models.

### **3. Key Takeaways: **
- Do not reprocess the data to ensure consistency with the results produced in the Report
- Running **`main.py`** is enough for the training, evaluation and applying the model to the data.


