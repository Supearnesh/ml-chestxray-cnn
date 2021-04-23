import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN architecture
class CNNClassifier(nn.Module):
    ### 5 layer architecture with 2 fully connected layers
    def __init__(self):
        super(CNNClassifier, self).__init__()
        ## Define layers of a CNN
        # convolutional layer (sees 112x160x1 tensor)
        self.conv_01 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # batch normalization applied to convolutional layer
        self.norm_01 = nn.BatchNorm2d(16)
        # convolutional layer (sees 56x80x16 tensor)
        self.conv_02 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # batch normalization applied to convolutional layer
        self.norm_02 = nn.BatchNorm2d(32)
        # convolutional layer (sees 28x40x32 tensor)
        self.conv_03 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # batch normalization applied to convolutional layer
        self.norm_03 = nn.BatchNorm2d(64)
        # convolutional layer pooled (sees 14x20x64 tensor)
        self.conv_04 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # batch normalization applied to convolutional layer
        self.norm_04 = nn.BatchNorm2d(128)
        # convolutional layer pooled (sees 7x10x128 tensor)
        self.conv_05 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # batch normalization applied to convolutional layer
        self.norm_05 = nn.BatchNorm2d(256)
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # linear layer (7 * 10 * 256 -> 2048)
        self.fc_01 = nn.Linear(7 * 10 * 256, 2048)
        # linear layer (2048 -> 2)
        self.fc_02 = nn.Linear(2048, 2)
        # dropout layer (p = 0.50)
        self.dropout = nn.Dropout(0.50)
    
    def forward(self, x):
        ## Define forward behavior
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.norm_01(self.conv_01(x))))
        x = self.pool(F.relu(self.norm_02(self.conv_02(x))))
        x = self.pool(F.relu(self.norm_03(self.conv_03(x))))
        x = self.pool(F.relu(self.norm_04(self.conv_04(x))))
        x = self.pool(F.relu(self.norm_05(self.conv_05(x))))
        # flatten image input
        x = x.view(-1, 7 * 10 * 256)
        # add dropout layer
        x = self.dropout(x)
        # add first hidden layer, with relu activation function
        x = F.relu(self.fc_01(x))
        # add dropout layer
        x = self.dropout(x)
        # add second hidden layer, with relu activation function
        x = self.fc_02(x)
        return x