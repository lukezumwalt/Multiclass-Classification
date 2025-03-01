'''
baseline.py

This CNN model is designed to accept a fixed-sized image and
pass it through a configurable number of both convolutional 
and fully-connected layers in order to satisfy a mutli-class 
classification.

It uses pytorch as its fundamental library, taking advantage
of its convolutional methods and neural net functioinality.

The main purpose of this class is for construction of the 
foundational model.  Regularization, training, and validation
are handled in separate class files.

Lukas Zumwalt
3/1/2025
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    '''
    Baseline Convolutional Neural Net.
    '''
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Adjust based on input image size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        Forward Pass of parameters through activation functions
        '''
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, (8, 8))  # Ensure feature map fits FC layer
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation here; use Softmax in loss function
        return x

# Example Usage
if __name__ == "__main__":
    model = BaselineCNN(num_classes=10)
    print(model)
