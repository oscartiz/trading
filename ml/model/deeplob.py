import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        # Inception module with 1x1, 1x3, 1x5 convolutions
        # We want the output of each branch to be out_channels.
        # Total output channels will be 3 * out_channels.
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding=(0, 2))
        
        # In the original paper they use LeakyReLU and MaxPooling within inception, 
        # but to keep it simple and effective as per our plan:
        self.bn = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x):
        # x is (B, in_channels, T, W)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        # Concatenate along the channel dimension
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        out = F.leaky_relu(out, 0.01)
        return out

class DeepLOB(nn.Module):
    def __init__(self, num_classes=3, conv_channels=(1, 32, 32, 32), inception_channels=64, lstm_hidden=64):
        """
        DeepLOB Architecture: CNN + Inception + LSTM
        Adapted for 40 features (10 levels * 4 features per level).
        """
        super(DeepLOB, self).__init__()
        
        # 1. Convolutional blocks to extract spatial features
        # Input shape: (B, 1, T, 40)
        
        self.conv1 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=(1, 2), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(conv_channels[1])
        # After conv1: W = 20
        
        self.conv2 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=(1, 2), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(conv_channels[2])
        # After conv2: W = 10
        
        self.conv3 = nn.Conv2d(conv_channels[2], conv_channels[3], kernel_size=(1, 10))
        self.bn3 = nn.BatchNorm2d(conv_channels[3])
        # After conv3: W = 1
        
        # 2. Inception modules
        self.inception1 = InceptionModule(conv_channels[3], inception_channels)
        # Output channels = 3 * 64 = 192
        self.inception2 = InceptionModule(inception_channels * 3, inception_channels)
        # Output channels = 192
        
        # 3. LSTM
        # Input to LSTM is (B, T, 192)
        lstm_input_size = inception_channels * 3
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden, num_layers=1, batch_first=True)
        
        # 4. Fully Connected Layer
        self.fc = nn.Linear(lstm_hidden, num_classes)
        
    def forward(self, x):
        # x shape: (B, 1, T, 40)
        
        # CNN blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.01)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.01)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.01)
        
        # Inception blocks
        x = self.inception1(x)
        x = self.inception2(x)
        
        # Prepare for LSTM
        # x is (B, 192, T, 1) -> squeeze W dim -> (B, 192, T) -> transpose to (B, T, 192)
        x = x.squeeze(3).transpose(1, 2)
        
        # LSTM
        # lstm_out is (B, T, hidden_size)
        # (h_n, c_n) where h_n is (num_layers, B, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output of the last time step
        # Equivalently: x = lstm_out[:, -1, :]
        x = h_n[-1] # shape (B, hidden_size)
        
        # Fully connected
        logits = self.fc(x)
        
        return logits
