import pandas as pd
import numpy as np
import logging
import asyncio
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import aiofiles
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_hierarchy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for model versioning and tracking."""
    name: str
    version: str
    type: str
    layer: str  # strategist, tactician, or executor
    features: List[str]
    parameters: Dict
    metrics: Dict
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        return cls(**data)

class AsyncModelLogger:
    """Asynchronous logging for model events and metrics."""
    
    def __init__(self, log_dir: Path = Path('logs/models')):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queue = asyncio.Queue()
        self.is_running = False
    
    async def start(self):
        """Start the logging loop."""
        self.is_running = True
        while self.is_running:
            try:
                log_entry = await self.queue.get()
                async with aiofiles.open(
                    self.log_dir / f"{log_entry['model']}_{log_entry['timestamp']:%Y%m%d}.log",
                    mode='a'
                ) as f:
                    await f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                logger.error(f"Logging error: {str(e)}")
    
    async def log(self, model: str, event_type: str, data: Dict):
        """Add a log entry to the queue."""
        await self.queue.put({
            'timestamp': datetime.now(),
            'model': model,
            'event_type': event_type,
            'data': data
        })
    
    async def stop(self):
        """Stop the logging loop."""
        self.is_running = False

class BaseModel:
    """Base class for all models in the hierarchy."""
    
    def __init__(self, name: str, metadata: ModelMetadata):
        self.name = name
        self.metadata = metadata
        self.logger = AsyncModelLogger()
        self.scaler = StandardScaler()
    
    async def log_event(self, event_type: str, data: Dict):
        """Log a model event asynchronously."""
        await self.logger.log(self.name, event_type, data)
    
    def save(self, path: Path):
        """Save model and metadata."""
        raise NotImplementedError
    
    def load(self, path: Path):
        """Load model and metadata."""
        raise NotImplementedError

class StrategistModel(BaseModel):
    """High-level strategic decision making."""
    
    def __init__(self, name: str, metadata: ModelMetadata):
        super().__init__(name, metadata)
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            loss_function='RMSE',
            verbose=False
        )
    
    async def train(self, X: pd.DataFrame, y: pd.Series, batch_size: int = 1000):
        """Train the strategist model with batch processing."""
        await self.log_event('training_start', {'shape': X.shape})
        
        try:
            for i in range(0, len(X), batch_size):
                X_batch = X.iloc[i:i+batch_size]
                y_batch = y.iloc[i:i+batch_size]
                
                if i == 0:  # First batch
                    self.model.fit(X_batch, y_batch)
                else:  # Subsequent batches
                    self.model.fit(X_batch, y_batch, init_model=self.model)
                
                await self.log_event('batch_complete', {
                    'batch': i // batch_size,
                    'samples': len(X_batch)
                })
            
            await self.log_event('training_complete', {
                'feature_importance': self.model.feature_importances_.tolist()
            })
            
        except Exception as e:
            await self.log_event('training_error', {'error': str(e)})
            raise

class TacticianModel(BaseModel):
    """Mid-level tactical decision making using LSTM."""
    
    def __init__(self, name: str, metadata: ModelMetadata, sequence_length: int = 20):
        super().__init__(name, metadata)
        self.sequence_length = sequence_length
        self.model = nn.LSTM(
            input_size=len(metadata.features),
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)
    
    def prepare_sequences(self, X: pd.DataFrame) -> torch.Tensor:
        """Prepare sequential data for LSTM."""
        sequences = []
        for i in range(len(X) - self.sequence_length):
            sequences.append(X.iloc[i:i+self.sequence_length].values)
        return torch.FloatTensor(np.array(sequences))
    
    async def train(self, X: pd.DataFrame, y: pd.Series, batch_size: int = 32):
        """Train the tactician model with sequence handling."""
        await self.log_event('training_start', {'shape': X.shape})
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            X_seq = self.prepare_sequences(pd.DataFrame(X_scaled, columns=X.columns))
            y_seq = torch.FloatTensor(y[self.sequence_length:].values)
            
            dataset = TensorDataset(X_seq, y_seq)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            optimizer = torch.optim.Adam(self.model.parameters())
            criterion = nn.MSELoss()
            
            for epoch in range(10):
                for batch_idx, (data, target) in enumerate(loader):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    lstm_out, _ = self.model(data)
                    predictions = self.fc(lstm_out[:, -1, :])
                    loss = criterion(predictions.squeeze(), target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    await self.log_event('batch_complete', {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': loss.item()
                    })
            
            await self.log_event('training_complete', {
                'final_loss': loss.item()
            })
            
        except Exception as e:
            await self.log_event('training_error', {'error': str(e)})
            raise

class ExecutorModel(BaseModel):
    """Low-level execution decision making using LightGBM."""
    
    def __init__(self, name: str, metadata: ModelMetadata):
        super().__init__(name, metadata)
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            objective='regression'
        )
    
    async def train(self, X: pd.DataFrame, y: pd.Series, batch_size: int = 1000):
        """Train the executor model with batch processing."""
        await self.log_event('training_start', {'shape': X.shape})
        
        try:
            for i in range(0, len(X), batch_size):
                X_batch = X.iloc[i:i+batch_size]
                y_batch = y.iloc[i:i+batch_size]
                
                if i == 0:  # First batch
                    self.model.fit(X_batch, y_batch)
                else:  # Update model
                    self.model.fit(
                        X_batch, y_batch,
                        init_model=self.model,
                        keep_training_booster=True
                    )
                
                await self.log_event('batch_complete', {
                    'batch': i // batch_size,
                    'samples': len(X_batch)
                })
            
            await self.log_event('training_complete', {
                'feature_importance': self.model.feature_importances_.tolist()
            })
            
        except Exception as e:
            await self.log_event('training_error', {'error': str(e)})
            raise

class ModelHierarchy:
    """Manages the hierarchical model system."""
    
    def __init__(self):
        self.strategist = None
        self.tactician = None
        self.executor = None
        self.logger = AsyncModelLogger()
    
    async def initialize_models(
        self,
        feature_lists: Dict[str, List[str]],
        model_params: Dict[str, Dict] = None
    ):
        """Initialize all models in the hierarchy."""
        model_params = model_params or {}
        
        # Initialize Strategist
        strategist_metadata = ModelMetadata(
            name="strategist_v1",
            version="1.0.0",
            type="catboost",
            layer="strategist",
            features=feature_lists.get('strategist', []),
            parameters=model_params.get('strategist', {}),
            metrics={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.strategist = StrategistModel("strategist_v1", strategist_metadata)
        
        # Initialize Tactician
        tactician_metadata = ModelMetadata(
            name="tactician_v1",
            version="1.0.0",
            type="lstm",
            layer="tactician",
            features=feature_lists.get('tactician', []),
            parameters=model_params.get('tactician', {}),
            metrics={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.tactician = TacticianModel("tactician_v1", tactician_metadata)
        
        # Initialize Executor
        executor_metadata = ModelMetadata(
            name="executor_v1",
            version="1.0.0",
            type="lightgbm",
            layer="executor",
            features=feature_lists.get('executor', []),
            parameters=model_params.get('executor', {}),
            metrics={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.executor = ExecutorModel("executor_v1", executor_metadata)
        
        await self.logger.log('hierarchy', 'initialization', {
            'strategist_features': len(feature_lists.get('strategist', [])),
            'tactician_features': len(feature_lists.get('tactician', [])),
            'executor_features': len(feature_lists.get('executor', []))
        })
    
    async def train_hierarchy(
        self,
        data: Dict[str, pd.DataFrame],
        targets: Dict[str, pd.Series],
        batch_sizes: Dict[str, int] = None
    ):
        """Train all models in the hierarchy."""
        batch_sizes = batch_sizes or {
            'strategist': 1000,
            'tactician': 32,
            'executor': 1000
        }
        
        try:
            # Train Strategist
            await self.strategist.train(
                data['strategist'],
                targets['strategist'],
                batch_sizes['strategist']
            )
            
            # Train Tactician
            await self.tactician.train(
                data['tactician'],
                targets['tactician'],
                batch_sizes['tactician']
            )
            
            # Train Executor
            await self.executor.train(
                data['executor'],
                targets['executor'],
                batch_sizes['executor']
            )
            
            await self.logger.log('hierarchy', 'training_complete', {
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            await self.logger.log('hierarchy', 'training_error', {
                'error': str(e)
            })
            raise
    
    def save_hierarchy(self, base_path: Path):
        """Save all models and their metadata."""
        base_path.mkdir(parents=True, exist_ok=True)
        self.strategist.save(base_path / 'strategist')
        self.tactician.save(base_path / 'tactician')
        self.executor.save(base_path / 'executor')
    
    def load_hierarchy(self, base_path: Path):
        """Load all models and their metadata."""
        self.strategist.load(base_path / 'strategist')
        self.tactician.load(base_path / 'tactician')
        self.executor.load(base_path / 'executor')

if __name__ == "__main__":
    # Example usage
    async def main():
        try:
            # Initialize hierarchy
            hierarchy = ModelHierarchy()
            
            # Define feature lists for each layer
            feature_lists = {
                'strategist': ['volatility', 'trend', 'volume_intensity'],
                'tactician': ['rsi', 'macd', 'bollinger'],
                'executor': ['spread', 'depth', 'momentum']
            }
            
            await hierarchy.initialize_models(feature_lists)
            
            # Load and prepare data
            df = pd.read_parquet('data/features/forex_features_aug2025.parquet')
            
            # Prepare data for each layer
            data = {
                'strategist': df[feature_lists['strategist']],
                'tactician': df[feature_lists['tactician']],
                'executor': df[feature_lists['executor']]
            }
            
            # Prepare targets (example using returns at different horizons)
            targets = {
                'strategist': df['returns'].shift(-60),  # 1-hour horizon
                'tactician': df['returns'].shift(-20),   # 20-min horizon
                'executor': df['returns'].shift(-1)      # 1-min horizon
            }
            
            # Train hierarchy
            await hierarchy.train_hierarchy(data, targets)
            
            # Save models
            hierarchy.save_hierarchy(Path('models/hierarchy'))
            
        except Exception as e:
            logger.error("Training failed", exc_info=True)
            raise
    
    asyncio.run(main())