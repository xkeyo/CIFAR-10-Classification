# Import the torch libraries for MLP training and testing functions
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Training function for all the MLP models
# model is the model to be trained
# train_features and train_labels are the training data
# number of training epoch is set to default to 20
# batch size is set to default to 32
# learning rate is set to default to 0.001
def train_mlp_model(model, train_features, train_labels, num_epochs=20, batch_size=32, learning_rate=0.001):
    # Create a TensorDataset and DataLoader for batch training
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss function set to cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer set to SGD with a momentum of 0.9 and learning rate of 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        running_loss = 0.0
        # Loop over the training data
        for features, labels in train_loader:
            optimizer.zero_grad() # Reset the gradients
            outputs = model(features) # Forward pass
            loss = criterion(outputs, labels) # Compute the loss between the output and the ground truth
            loss.backward() # backward pass
            optimizer.step() # Update the weights
            running_loss += loss.item() # Add the loss to the running loss

        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Prediction function used for the trained MLP models
def predict_mlp(model, test_features):
    # Set the model to evaluation mode
    model.eval()
    # Loop over the test data
    with torch.no_grad():
        outputs = model(test_features) # Forward pass
        _, predicted = torch.max(outputs, 1) # Get the predicted class
    return predicted
