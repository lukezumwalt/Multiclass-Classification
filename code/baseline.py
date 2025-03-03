'''
baseline.py

This CNN model is designed to accept fixed-sized images and
pass them through a configurable number of both convolutional 
and fully-connected layers in order to satisfy a mutli-class 
classification.

It uses pytorch as its fundamental library, taking advantage
of its convolutional methods and neural net functioinality.

Leaky ReLU is the primary activation function for neruons in
this model, and softmax is used for the final layer.

The main purpose of this class is for construction of the 
foundational model.  Rgularization, training, validation,
and data access are handled in separate class files.

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
    def __init__(self, num_classes=5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        '''
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2)
        '''

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 256)  # Adjust based on input image size
        self.fc2 = nn.Linear(256, 128) # second term is is effectively how discretized the output is
        self.fc3 = nn.Linear(128, num_classes)

        # Activation function
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        '''
        Forward Pass of parameters through activation functions,
        excepting the final fully-connected layer.
        '''
        # Convolutional Layers:
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.pool(x)

        # Formatting for FC, no learning here:
        x = F.adaptive_avg_pool2d(x, (8, 8))  # Ensure feature map fits FC layer
        x = torch.flatten(x, start_dim=1)

        # Fully Connected Layers:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation here; use Softmax in loss function

        return x

# Example Usage
if __name__ == "__main__":
    model = BaselineCNN(num_classes=5)
    print(model)
