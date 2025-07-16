import torch
import torch.nn as nn
import torchvision.models as models

class GenreClassifierResNet(nn.Module):
    def __init__(self, num_genres=10, pretrained=True):
        super(GenreClassifierResNet, self).__init__()
        # Load a pre-trained ResNet18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # 1. Modify the first convolutional layer to accept 1-channel (grayscale) input
        # Original conv1: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias
        )
        
        # If using pre-trained weights, adapt them for single-channel input
        if pretrained:
            # Average the weights of the original 3-channel conv layer across the channel dimension
            original_weights = original_conv1.weight.data
            new_weights = original_weights.mean(dim=1, keepdim=True)
            self.resnet.conv1.weight.data = new_weights

        # 2. Modify the final fully connected layer for our number of genres
        # Original fc: Linear(in_features=512, out_features=1000, bias=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_genres)

    def forward(self, x):
        # The ResNet model will handle the entire forward pass.
        return self.resnet(x)

# For compatibility with the existing training script, we can alias the new model
# This avoids having to change the model instantiation code in train.py
GenreClassifierCNN = GenreClassifierResNet
