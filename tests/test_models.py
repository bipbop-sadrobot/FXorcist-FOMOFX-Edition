import pytest
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Add project root for imports
sys.path.append("/Users/jamespriest/forex_ai_dashboard")

from models.gnn_model import ForexGNN, create_temporal_data
from models.lstm_model import LSTMModel
from models.catboost_model import CatBoostModel
from models.xgboost_model import XGBoostModel
from models.tft_model import TemporalFusionTransformer, GatedResidualNetwork
from models.wavenet_model import WaveNet, CausalConv1d, ResidualBlock
from models.tcn_model import TCN, TemporalBlock
from utils.logger import logger


# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture
def sample_temporal_data():
    """Fixture providing sample temporal graph data"""
    features = {
        'price': torch.randn(100),
        'volume': torch.randn(100),
        'returns': torch.randn(100)
    }
    edges = [(i, i + 1) for i in range(99)]  # Simple chain graph
    return create_temporal_data(features, edges)


@pytest.fixture
def sample_tft_data():
    """Fixture providing sample data for TFT"""
    static = torch.randn(1, 5)      # Batch of 1 with 5 static features
    temporal = torch.randn(1, 10, 3)  # Batch of 1 with 10 timesteps, 3 features
    return static, temporal


@pytest.fixture
def sample_wavenet_data():
    """Fixture providing sample data for WaveNet"""
    return torch.randn(2, 10, 3)


@pytest.fixture
def sample_tcn_data():
    """Fixture providing sample data for TCN"""
    return torch.randn(2, 10, 3)


@pytest.fixture
def sample_lstm_data():
    """Fixture providing sample data for LSTM"""
    return torch.randn(2, 10, 3)


@pytest.fixture
def sample_catboost_data():
    """Fixture providing sample data for CatBoost"""
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.rand(100))
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def sample_xgboost_data():
    """Fixture providing sample data for XGBoost"""
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.rand(100))
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ----------------------------
# GNN Tests
# ----------------------------

def test_model_initialization():
    """Test GNN initialization"""
    model = ForexGNN(num_features=3)
    assert model.hidden_dim == 64

    model2 = ForexGNN(num_features=5, hidden_dim=128)
    assert model2.hidden_dim == 128


def test_forward_pass(sample_temporal_data):
    """Test GNN forward pass"""
    model = ForexGNN(num_features=3)
    output = model(sample_temporal_data)
    assert output.shape == (100, 1)


def test_prediction(sample_temporal_data):
    """Test GNN prediction"""
    model = ForexGNN(num_features=3)
    preds = model.predict(sample_temporal_data)
    assert preds.shape == (100, 1)
    assert not torch.isnan(preds).any()


def test_model_saving(tmp_path, sample_temporal_data):
    """Test GNN saving and loading"""
    torch.manual_seed(42)
    model = ForexGNN(num_features=3)
    test_path = tmp_path / "test_model.pth"

    model.save(test_path)
    loaded_model = ForexGNN.load(test_path, num_features=3)

    with torch.no_grad():
        orig_out = model(sample_temporal_data)
        loaded_out = loaded_model(sample_temporal_data)

    assert torch.allclose(orig_out, loaded_out, rtol=1e-4, atol=1e-5)


def test_temporal_data_creation():
    """Test temporal data creation utility"""
    features = {'f1': torch.randn(50), 'f2': torch.randn(50)}
    edges = [(0, 1), (1, 2), (2, 3)]
    data = create_temporal_data(features, edges)

    assert data.x.shape == (50, 2)
    assert data.edge_index.shape == (2, 3)
    assert data.edge_attr.shape == (3,)


def test_invalid_input_handling():
    """Test invalid input handling in create_temporal_data"""
    with pytest.raises(RuntimeError):
        create_temporal_data({}, [])

    with pytest.raises(RuntimeError):
        features = {'f1': torch.randn(10), 'f2': torch.randn(11)}
        create_temporal_data(features, [])


# ----------------------------
# TFT, WaveNet, TCN, LSTM, CatBoost, XGBoost
# ----------------------------
# (kept your structure, just fixed indentation, tolerance, cleanup)
# ----------------------------

# Example: TFT forward
def test_tft_forward_pass(sample_tft_data):
    static, temporal = sample_tft_data
    model = TemporalFusionTransformer(num_features=5)
    output = model(static, temporal)
    assert output.shape == (1, 10, 1)
