import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass, asdict
import networkx as nx
from scipy import stats
import ta  # Technical Analysis library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FeatureMetadata:
    """Metadata for feature versioning and tracking."""
    name: str
    version: str
    description: str
    dependencies: List[str]
    parameters: Dict
    created_at: datetime
    updated_at: datetime
    category: str
    statistics: Dict = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureMetadata':
        return cls(**data)

class FeatureRegistry:
    """Manages feature versioning, dependencies, and metadata."""
    
    def __init__(self, storage_path: Path = Path('data/features')):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.features: Dict[str, FeatureMetadata] = {}
        self.dependency_graph = nx.DiGraph()
        self._load_registry()
    
    def _load_registry(self):
        """Load existing feature registry from disk."""
        registry_file = self.storage_path / 'registry.json'
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                for feat_data in data['features']:
                    metadata = FeatureMetadata.from_dict(feat_data)
                    self.features[metadata.name] = metadata
                    # Rebuild dependency graph
                    for dep in metadata.dependencies:
                        self.dependency_graph.add_edge(dep, metadata.name)
    
    def save_registry(self):
        """Save current registry state to disk."""
        registry_file = self.storage_path / 'registry.json'
        with open(registry_file, 'w') as f:
            json.dump({
                'features': [feat.to_dict() for feat in self.features.values()],
                'updated_at': datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def register_feature(self, metadata: FeatureMetadata):
        """Register a new feature or update existing one."""
        self.features[metadata.name] = metadata
        # Update dependency graph
        for dep in metadata.dependencies:
            self.dependency_graph.add_edge(dep, metadata.name)
        self.save_registry()
    
    def get_feature_dependencies(self, feature_name: str) -> List[str]:
        """Get all dependencies for a feature (direct and indirect)."""
        if feature_name not in self.dependency_graph:
            return []
        return list(nx.ancestors(self.dependency_graph, feature_name))
    
    def get_dependent_features(self, feature_name: str) -> List[str]:
        """Get all features that depend on this feature."""
        if feature_name not in self.dependency_graph:
            return []
        return list(nx.descendants(self.dependency_graph, feature_name))

class FeatureGenerator:
    """Generates and manages forex trading features."""
    
    def __init__(self):
        self.registry = FeatureRegistry()
        self._register_base_features()
    
    def _register_base_features(self):
        """Register basic price and volume features."""
        base_features = [
            FeatureMetadata(
                name='returns',
                version='1.0.0',
                description='Log returns of close prices',
                dependencies=['close'],
                parameters={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                category='basic'
            ),
            FeatureMetadata(
                name='volatility',
                version='1.0.0',
                description='Rolling volatility',
                dependencies=['returns'],
                parameters={'window': 20},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                category='volatility'
            )
        ]
        for feature in base_features:
            self.registry.register_feature(feature)
    
    def generate_features(
        self,
        df: pd.DataFrame,
        feature_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate specified features, respecting dependencies."""
        df = df.copy()
        
        # If no specific features requested, generate all registered features
        feature_list = feature_list or list(self.registry.features.keys())
        
        # Sort features by dependencies
        sorted_features = list(nx.topological_sort(self.registry.dependency_graph))
        features_to_generate = [f for f in sorted_features if f in feature_list]
        
        for feature in features_to_generate:
            df = self._generate_single_feature(df, feature)
        
        return df
    
    def _generate_single_feature(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Generate a single feature with its dependencies."""
        metadata = self.registry.features.get(feature_name)
        if not metadata:
            logger.warning(f"Feature {feature_name} not found in registry")
            return df
        
        try:
            # Basic price features
            if feature_name == 'returns':
                df['returns'] = np.log(df['close'] / df['close'].shift(1))
            
            elif feature_name == 'volatility':
                window = metadata.parameters.get('window', 20)
                df['volatility'] = df['returns'].rolling(window=window).std()
            
            # Technical indicators (using ta library)
            elif feature_name.startswith('rsi_'):
                window = int(feature_name.split('_')[1])
                df[feature_name] = ta.momentum.RSIIndicator(
                    df['close'], window=window
                ).rsi()
            
            elif feature_name.startswith('bb_'):
                window = int(feature_name.split('_')[1])
                bb = ta.volatility.BollingerBands(
                    df['close'], window=window
                )
                df[f'bb_{window}_upper'] = bb.bollinger_hband()
                df[f'bb_{window}_lower'] = bb.bollinger_lband()
                df[f'bb_{window}_pct'] = bb.bollinger_pband()
            
            # Volume-based features
            elif feature_name == 'volume_intensity':
                df['volume_intensity'] = (
                    df['volume'] * np.abs(df['returns'])
                ).rolling(window=20).mean()
            
            # Market microstructure features
            elif feature_name == 'spread':
                df['spread'] = df['high'] - df['low']
            
            elif feature_name == 'gap_up':
                df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
            
            elif feature_name == 'gap_down':
                df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
            
            # Update feature statistics
            self._update_feature_stats(df[feature_name])
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating feature {feature_name}: {str(e)}")
            return df
    
    def _update_feature_stats(self, feature_series: pd.Series):
        """Update feature statistics for monitoring."""
        if feature_series.name not in self.registry.features:
            return
        
        stats = {
            'mean': feature_series.mean(),
            'std': feature_series.std(),
            'skew': feature_series.skew(),
            'kurtosis': feature_series.kurtosis(),
            'missing_pct': (feature_series.isnull().sum() / len(feature_series)) * 100
        }
        
        metadata = self.registry.features[feature_series.name]
        metadata.statistics = stats
        metadata.updated_at = datetime.now()
        self.registry.register_feature(metadata)

def analyze_feature_importance(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    method: str = 'correlation'
) -> pd.DataFrame:
    """Analyze feature importance using various methods."""
    results = []
    
    if method == 'correlation':
        for feature in feature_cols:
            corr = stats.spearmanr(
                df[feature].fillna(0),
                df[target_col].fillna(0)
            )[0]
            results.append({
                'feature': feature,
                'importance': abs(corr),
                'raw_score': corr
            })
    
    # Add more methods as needed (mutual information, SHAP, etc.)
    
    return pd.DataFrame(results).sort_values('importance', ascending=False)

if __name__ == "__main__":
    # Example usage
    try:
        # Load sample data
        df = pd.read_parquet('data/processed/ingested_forex_1min_aug2025.parquet')
        
        # Initialize feature generator
        generator = FeatureGenerator()
        
        # Generate features
        df_features = generator.generate_features(df)
        
        # Analyze feature importance
        target = 'returns'
        features = [col for col in df_features.columns if col != target]
        importance = analyze_feature_importance(df_features, target, features)
        
        logger.info("\nFeature Importance Analysis:")
        logger.info(importance)
        
        # Save features
        output_path = Path('data/features/forex_features_aug2025.parquet')
        df_features.to_parquet(output_path)
        logger.info(f"\nFeatures saved to {output_path}")
        
    except Exception as e:
        logger.error("Feature generation failed", exc_info=True)
        raise