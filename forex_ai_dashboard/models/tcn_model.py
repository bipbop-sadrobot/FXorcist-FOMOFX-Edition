import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from ..utils.logger import logger


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution and residual connection"""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)

class TCN(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self,
                 num_features: int,
                 num_channels: List[int],
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.num_features = num_features
        self.num_channels = num_channels
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout
                )
            ]
        
        # Skip connections from early to late layers
        self.skip_connections = nn.ModuleList()
        for i in range(1, num_levels):
            if i % 2 == 0:  # Add skip connection every 2 layers
                self.skip_connections.append(
                    nn.Conv1d(num_channels[i-2], num_channels[i], 1)
                )
        
        self.network = nn.ModuleList(layers)
        self.output = nn.Linear(num_channels[-1], 1)
        
        logger.info(f"Initialized TCN with {num_levels} layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info(f"Input shape: {x.shape}")
        # Input shape: (batch, time, features)
        x = x.permute(0, 2, 1)  # (batch, features, time)
        
        # Process through network with skip connections
        skip_outputs = []
        for i, layer in enumerate(self.network):
            x = layer(x)
            # Add skip connections from earlier layers
            if i >= 2 and i % 2 == 0:
                skip_idx = (i // 2) - 1
                if skip_idx < len(self.skip_connections):
                    skip = self.skip_connections[skip_idx](skip_outputs[i-2])
                    x = x + skip  # Residual connection
            skip_outputs.append(x)
        
        # Use final layer output
        out = x.permute(0, 2, 1)  # (batch, time, features)
        return self.output(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the model"""
        with torch.no_grad():
            return self.forward(x)

    def save(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)
        logger.info(f"Saved TCN model weights to {path}")

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from file"""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"Loaded TCN model from {path}")
        return model
