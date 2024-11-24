import torch.nn as nn

# Variant 3: Changing the size of the hidden layers of base MLP model to a bigger size 
class MLP(nn.Module):
    # Initialize the model
    # Input layer is 50
    # Hidden layer 1 is 1024 with ReLU activation
    # Hidden layer 2 is 1024 with ReLU activation and Batch Normalization
    # Output layer is 10
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(50, 1024)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(1024, 1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(1024, 10)

    # Forward pass through the model
    # Input -> Layer 1 -> ReLU -> Layer 2 -> BatchNorm -> ReLU -> Layer 3 -> Output.
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x
