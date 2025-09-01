import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric_temporal.nn.recurrent.temporalgcn import TGCN
from torch_geometric.data import TemporalData
from typing import List, Dict
from ..utils.logger import logger

class ForexGNN(nn.Module):
    def __init__(self, 
                 num_features: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList([
            TGCN(in_channels=hidden_dim, out_channels=hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Graph attention layers with dynamic head configuration
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim, 
                hidden_dim, 
                heads=min(num_heads, hidden_dim // 4),  # Dynamic head count
                dropout=dropout,
                add_self_loops=False,  # More efficient for temporal graphs
                attention_dropout=dropout  # Additional attention dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization for attention outputs
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Initialized ForexGNN with {num_layers} layers")

    def forward(self, data: TemporalData) -> torch.Tensor:
        x, edge_index, edge_weight, time = data.x, data.edge_index, data.edge_attr, data.t
        logger.info(f"Input x shape: {x.shape}")
        # Project input features
        x = self.input_proj(x)
        logger.info(f"Shape after input projection: {x.shape}")
        logger.info(f"Shape before reshaping: {x.shape}")
        self.num_nodes = x.shape[0]
        logger.info(f"num_nodes: {self.num_nodes}, num_features: {self.num_features}, hidden_dim: {self.hidden_dim}")
        # Temporal and graph processing
        for i, (temp_conv, gat_layer) in enumerate(zip(self.temporal_convs, self.gat_layers)):
            # Temporal convolution
            x = temp_conv(x, edge_index)
            
            # Graph attention with layer norm
            x_gat = gat_layer(x, edge_index, edge_weight)
            x_gat = self.layer_norms[i](x_gat)  # Layer normalization
            
            # More efficient attention aggregation
            x_gat = x_gat.view(-1, self.num_heads, self.hidden_dim)
            x_gat = x_gat.mean(dim=1)  # Mean over heads
            
            # Residual connection with learned weights
            gate = torch.sigmoid(self.dropout(x_gat))
            x = x * gate + x_gat * (1 - gate)  # Gated residual
        
        # Final projection
        out = self.output_proj(x)
        return out

    def predict(self, data: TemporalData) -> torch.Tensor:
        """Make predictions with the model"""
        with torch.no_grad():
            return self.forward(data)

    def save(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)
        logger.info(f"Saved model weights to {path}")

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from file"""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"Loaded model from {path}")
        return model

def create_temporal_data(features: Dict[str, torch.Tensor], 
                        edges: List[tuple]) -> TemporalData:
    """Create temporal graph data from features and edges"""
    x = torch.stack(list(features.values()), dim=1)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.ones(edge_index.size(1))
    time = torch.arange(x.size(0))
    return TemporalData(x=x, edge_index=edge_index, edge_attr=edge_weight, t=time)
