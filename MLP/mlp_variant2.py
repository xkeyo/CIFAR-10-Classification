import torch.nn as nn

# Variant 2: Adding layers to the MLP base model
class MLP(nn.Module):
    # Initialize the model
    # Input layer is 50
    # Hidden layer 1 is 512 with ReLU activation
    # Hidden layer 2 is 512 with BatchNorm and ReLU activation
    # Hidden layer 3 is 512 with BatchNorm and ReLU activation
    # Output layer is 10
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(50, 512)
        self.relu1 = nn.ReLU()

        self.layer2 = nn.Linear(512, 512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()

        self.layer3 = nn.Linear(512, 512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()

        self.layer4 = nn.Linear(512, 10)

    
    # Forward pass through the model
    # Input -> Hidden 1 -> ReLU -> Hidden 2 -> BatchNorm -> ReLU -> Hidden 3 -> BatchNorm -> ReLU -> Output
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        return x
