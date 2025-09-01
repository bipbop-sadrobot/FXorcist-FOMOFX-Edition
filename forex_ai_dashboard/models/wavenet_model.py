import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from ..utils.logger import logger

class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class ResidualBlock(nn.Module):
    """WaveNet residual block with gated activation"""
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = CausalConv1d(
            residual_channels, residual_channels, kernel_size, dilation
        )
        self.gate_conv = CausalConv1d(
            residual_channels, residual_channels, kernel_size, dilation
        )
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        self.dilation = dilation

    def forward(self, x):
        # Gated activation
        filter = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        activation = filter * gate

        # Residual and skip connections
        residual = self.residual_conv(activation)
        skip = self.skip_conv(activation)
        return (x + residual) * 0.7071, skip  # Scale residual for stability

class WaveNet(nn.Module):
    def __init__(self,
                 num_features: int,
                 residual_channels: int = 64,
                 skip_channels: int = 64,
                 num_layers: int = 10,
                 output_dim: int = 1,
                 kernel_size: int = 3):
        super().__init__()
        self.num_features = num_features
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.output_dim = output_dim

        # Input convolution
        self.input_conv = CausalConv1d(num_features, residual_channels, 1)

        # Residual blocks with exponential dilation
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                residual_channels,
                skip_channels,
                kernel_size,
                dilation=2 ** (i % 10)  # Cycle through 1,2,4...512
            )
            for i in range(num_layers)
        ])

        # Output layers
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, output_dim, 1)
        )

        logger.info(f"Initialized WaveNet with {num_layers} layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info(f"Input shape: {x.shape}")
        # Input processing
        x = x.permute(0, 2, 1)  # (batch, features, time)
        x = self.input_conv(x)
        
        # Residual blocks
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Output processing
        out = torch.sum(torch.stack(skip_connections), dim=0)
        out = self.output_conv(out)
        return out.permute(0, 2, 1)  # (batch, time, features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the model"""
        with torch.no_grad():
            return self.forward(x)

    def save(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)
        logger.info(f"Saved WaveNet model weights to {path}")

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from file"""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"Loaded WaveNet model from {path}")
        return model
