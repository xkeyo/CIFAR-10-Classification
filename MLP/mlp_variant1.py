import torch.nn as nn

# Variant 1: Removing layers from the MLP base model 
class MLP(nn.Module):
    # Initialize the model
    # Input layer is 50
    # Hidden layer 1 is 512 with ReLU activation
    # Output layer is 10
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(50, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 10)  # Directly connect to output layer

    # Forward pass
    # Input layer -> Layer 1 -> ReLU -> Layer 2 -> Output.
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x
