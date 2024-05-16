import torch.nn as nn

class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Head, self).__init__()
        self.num_classes = num_classes
        
        # Convolutional layers for heatmap prediction
        self.conv_heatmap = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        
        # Convolutional layers for width prediction
        self.conv_width = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Convolutional layers for height prediction
        self.conv_height = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Convolutional layers for offset prediction
        self.conv_offset = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # print(x.shape)
        print(self.num_classes)
        heatmap = self.conv_heatmap(x)
        width = self.conv_width(x)
        height = self.conv_height(x)
        offset = self.conv_offset(x)
        
        heatmap = self.sigmoid(heatmap)
        width = self.relu(width)
        height = self.relu(height)
        
        return heatmap, width, height, offset