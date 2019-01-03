## define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
            
        # Convolutional layers
        # 1 input image  1 channel , 32 output feature maps, 5x5  conv kernel
        
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for image : (32, 220, 220)
        # after  pool  (32, 110, 110)
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # 32 input feature maps, 64 output feature maps, 3x3 square conv kernel
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # the output Tensor for image : (64, 108, 108)
        # after one pool  (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # 64 input feature maps, 128 output feature maps, 3x3  conv kernel
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output Tensor for image : (128, 52, 52)
        # after pool  (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # 128 input feature maps, 256 feature maps, 1x1 conv kernel
        ## output size = (W-F)/S +1 = (26-1)/1 +1 = 26
        # the output Tensor for image : (256, 26, 26)
        # after pool layer (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # dropout Layer for solve overfit problem
        
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.4)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # 256 outputs * 13*13 filtered/pooled map
        self.fc1 = nn.Linear(256*13*13, 13*136)
        
        # at the end, 136 output channels (2 for each of the 68 keypoint (x, y) pairs)
        self.fc2 = nn.Linear(13*136, 136)
        

        
    def forward(self, x):
        
        # four conv/relu/pool layers
        
        x = self.dropout1(self.pool(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool(F.elu(self.conv2(x))))
        x = self.dropout3(self.pool(F.elu(self.conv3(x))))
        x = self.dropout4(self.pool(F.elu(self.conv4(x))))
        
        # linear layer preparation
        
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.elu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x