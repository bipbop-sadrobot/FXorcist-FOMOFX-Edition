import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from ..utils.logger import logger

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for variable selection"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        if input_dim != hidden_dim:
            self.residual = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.dropout(h)
        gate = torch.sigmoid(self.gate(x))
        res = self.residual(x)
        return self.layer_norm(gate * h + (1 - gate) * res)

class TemporalSelfAttention(nn.Module):
    """Temporal Self-Attention Layer"""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        return self.layer_norm(attn_out + x)

class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 seq_len: int = 10,
                 temporal_feat_dim: int = 3):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.temporal_feat_dim = temporal_feat_dim
        # Input projections
        self.input_projection = nn.Linear(num_features, hidden_dim)
        self.temporal_projection = nn.Linear(temporal_feat_dim, hidden_dim)

        # Variable selection networks
        self.static_selector = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        self.temporal_selector = GatedResidualNetwork(seq_len * temporal_feat_dim, hidden_dim, dropout)
        
        # Static encoders
        self.static_encoder = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        self.static_context = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal processing
        self.temporal_encoder = nn.ModuleList([
            TemporalSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"Initialized TFT with {num_layers} layers")

    def forward(self, 
               static_features: torch.Tensor,
               past_features: torch.Tensor,
               future_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        logger.info(f"static_features shape: {static_features.shape}")
        logger.info(f"past_features shape: {past_features.shape}")

        static_features = self.input_projection(static_features)
        
        # Reshape temporal features for processing
        batch_size = past_features.size(0)
        past_features = past_features.reshape(batch_size, self.seq_len, -1)
        
        # Project temporal features
        past_features = self.temporal_projection(past_features)

        # Static variable selection
        static_vars = self.static_selector(static_features)
        
        # Temporal variable selection
        temporal_vars = past_features  # Skip flattening to preserve temporal dim
        
        # Static context generation
        static_encoded = self.static_encoder(static_vars)
        static_context = self.static_context(static_encoded)
        
        # Temporal processing
        temporal_out = temporal_vars
        for layer in self.temporal_encoder:
            temporal_out = layer(temporal_out, static_context)
        
        # Output
        predictions = self.output_layer(temporal_out)
        return predictions

    def predict(self, 
               static_features: torch.Tensor,
               past_features: torch.Tensor) -> torch.Tensor:
        """Make predictions with the model"""
        with torch.no_grad():
            return self.forward(static_features, past_features)

    def save(self, path: str):
        """Save model weights"""
        self.eval()  # Ensure consistent behavior during save
        torch.save(self.state_dict(), path)
        logger.info(f"Saved TFT model weights to {path}")

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from file"""
        if 'num_layers' not in kwargs:
            raise ValueError("num_layers must be specified when loading TFT model")
            
        # Create model with specified parameters
        model = cls(
            num_features=kwargs['num_features'],
            hidden_dim=kwargs.get('hidden_dim', 64),
            num_heads=kwargs.get('num_heads', 4),
            num_layers=kwargs['num_layers'],
            output_dim=kwargs.get('output_dim', 1),
            seq_len=kwargs.get('seq_len', 10),
            temporal_feat_dim=kwargs.get('temporal_feat_dim', 3)
        )
        
        # Load state dict with strict checking
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        logger.info(f"Loaded TFT model from {path}")
        return model
