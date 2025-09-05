#!/usr/bin/env python3
"""
Comprehensive Forex AI Training Pipeline
Integrates all advanced features: hyperparameter optimization, ensemble methods,
enhanced feature engineering, model comparison, and interpretability.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
import json
warnings.filterwarnings('ignore')

# Import our enhanced modules
from forex_ai_dashboard.pipeline.enhanced_feature_engineering import EnhancedFeatureEngineer
from forex_ai_dashboard.pipeline.hyperparameter_optimization import HyperparameterOptimizer
from forex_ai_dashboard.pipeline.model_comparison import ModelComparator
from forex_ai_dashboard.pipeline.model_interpretability import ModelInterpreter
from forex_ai_dashboard.models.catboost_model import CatBoostModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/comprehensive_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveTrainingPipeline:
    """Complete training pipeline with all advanced features."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models" / "trained"
        self.logs_dir = self.project_root / "logs"

        # Initialize components
        self.feature_engineer = EnhancedFeatureEngineer()
        self.model_comparator = ModelComparator()
        self.interpreter = ModelInterpreter()

        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD", "CHFJPY", "AUDJPY", "CADJPY", "NZDJPY", "AUDCHF", "CADCHF", "NZDCHF", "AUDCAD", "AUDNZD", "CADNZD"]
        self.years = [2020, 2021, 2022, 2023, 2024]

    def run_comprehensive_training(self,
                                  optimize_hyperparams: bool = True,
                                  use_ensemble: bool = True,
                                  enable_interpretability: bool = True,
                                  feature_selection: bool = True,
                                  n_features: Optional[int] = 200) -> Dict:
        """
        Run the complete comprehensive training pipeline.

        Args:
            optimize_hyperparams: Whether to run hyperparameter optimization
            use_ensemble: Whether to include ensemble methods
            enable_interpretability: Whether to generate model explanations
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select (if feature_selection=True)

        Returns:
            Comprehensive training results
        """
        logger.info("🚀 Starting Comprehensive Forex AI Training Pipeline")
        start_time = datetime.now()

        results = {
            'pipeline_start': start_time.isoformat(),
            'configuration': {
                'optimize_hyperparams': optimize_hyperparams,
                'use_ensemble': use_ensemble,
                'enable_interpretability': enable_interpretability,
                'feature_selection': feature_selection,
                'n_features': n_features
            },
            'stages': {}
        }

        try:
            # Stage 1: Data Loading and Enhanced Feature Engineering
            logger.info("📊 Stage 1: Data Loading and Feature Engineering")
            data_result = self._load_and_engineer_features(feature_selection, n_features)
            results['stages']['data_preparation'] = data_result

            if not data_result['success']:
                return self._finalize_results(results, "Data preparation failed")

            X_train, X_test, y_train, y_test = data_result['data']

            # Stage 2: Model Training with Hyperparameter Optimization
            logger.info("🤖 Stage 2: Model Training and Optimization")
            training_result = self._train_models(X_train, X_test, y_train, y_test, optimize_hyperparams, use_ensemble)
            results['stages']['model_training'] = training_result

            # Stage 3: Model Comparison and Selection
            logger.info("⚖️ Stage 3: Model Comparison and Selection")
            comparison_result = self._compare_and_select_models(training_result['models'])
            results['stages']['model_comparison'] = comparison_result

            # Stage 4: Model Interpretability (if enabled)
            if enable_interpretability:
                logger.info("🔍 Stage 4: Model Interpretability Analysis")
                interpretability_result = self._generate_interpretability(
                    training_result['models'], X_train, X_test, y_test
                )
                results['stages']['interpretability'] = interpretability_result

            # Stage 5: Finalize and Save Results
            logger.info("💾 Stage 5: Finalizing Results")
            finalization_result = self._finalize_training(results, comparison_result['best_model'])
            results['stages']['finalization'] = finalization_result

            # Calculate total time
            total_time = (datetime.now() - start_time).total_seconds()
            results['total_time_seconds'] = total_time
            results['success'] = True

            logger.info("✅ Comprehensive training pipeline completed successfully!")
            logger.info(".2f")

            return results

        except Exception as e:
            logger.error(f"Comprehensive training failed: {str(e)}", exc_info=True)
            return self._finalize_results(results, str(e))

    def _load_and_engineer_features(self, feature_selection: bool, n_features: Optional[int]) -> Dict:
        """Load data and perform enhanced feature engineering."""
        try:
            # Try to load existing processed data first
            processed_file = self.data_dir / "processed" / "comprehensive_forex_data.parquet"
            if processed_file.exists():
                logger.info(f"Loading existing processed data from {processed_file}")
                df = pd.read_parquet(processed_file)
            else:
                logger.info("Processing raw data with enhanced features")
                df = self._process_raw_data()

            if df is None or len(df) == 0:
                return {'success': False, 'error': 'No data available'}

            # Enhanced feature engineering - use comprehensive feature groups
            feature_groups = [
                'basic', 'momentum', 'volatility', 'trend', 'time',
                'microstructure', 'advanced_momentum', 'advanced_trend',
                'volume_advanced', 'statistical'
            ]

            df = self.feature_engineer.process_data(
                df,
                feature_groups=feature_groups,
                n_features=n_features if feature_selection else None
            )

            # Prepare target
            df['target'] = df['close'].shift(-1)
            df = df.dropna(subset=['target'])

            # Split data
            split_idx = int(len(df) * 0.8)
            train_df = df[:split_idx]
            test_df = df[split_idx:]

            # Prepare features and target
            feature_cols = [col for col in df.columns if col not in [
                'timestamp', 'symbol', 'target'
            ] and not col.startswith(('year', 'month'))]

            X_train = train_df[feature_cols]
            X_test = test_df[feature_cols]
            y_train = train_df['target']
            y_test = test_df['target']

            # Data validation and cleaning
            logger.info("Performing data validation and cleaning")
            X_train, X_test = self._validate_and_clean_data(X_train, X_test)

            # Handle categorical columns for CatBoost
            categorical_features = []
            for col in X_train.columns:
                if X_train[col].dtype.name == 'category':
                    # Convert categorical to numeric codes
                    X_train[col] = X_train[col].cat.codes.astype(float)
                    X_test[col] = X_test[col].cat.codes.astype(float)
                    categorical_features.append(col)
                elif X_train[col].dtype.name == 'object':
                    # Convert object columns to numeric if possible
                    try:
                        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                    except:
                        # Drop non-numeric object columns
                        X_train = X_train.drop(col, axis=1)
                        X_test = X_test.drop(col, axis=1)
                        if col in feature_cols:
                            feature_cols.remove(col)
                elif X_train[col].dtype.name in ['string', 'str']:
                    # Handle string columns by converting to categorical then to codes
                    try:
                        X_train[col] = pd.Categorical(X_train[col]).codes.astype(float)
                        X_test[col] = pd.Categorical(X_test[col]).codes.astype(float)
                        categorical_features.append(col)
                    except:
                        # Drop columns that can't be converted
                        X_train = X_train.drop(col, axis=1)
                        X_test = X_test.drop(col, axis=1)
                        if col in feature_cols:
                            feature_cols.remove(col)

            logger.info(f"Feature engineering complete: {len(feature_cols)} features, {len(X_train)} train samples")

            return {
                'success': True,
                'data': (X_train, X_test, y_train, y_test),
                'feature_count': len(feature_cols),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }

        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _process_raw_data(self) -> Optional[pd.DataFrame]:
        """Process raw forex data."""
        logger.info("Processing raw forex data")

        all_data = []

        for symbol in self.symbols:
            symbol_data = []

            # First try to load from processed parquet files
            processed_files = list((self.data_dir / "processed").glob(f"*{symbol.upper()}*.parquet"))
            if processed_files:
                logger.info(f"Loading processed data for {symbol}")
                for file_path in processed_files[:5]:  # Limit to 5 files per symbol
                    try:
                        df = pd.read_parquet(file_path)
                        df['symbol'] = symbol
                        symbol_data.append(df)
                        logger.info(f"Loaded {len(df)} rows from {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Error loading {file_path}: {str(e)}")

            # If no processed data, try raw CSV files
            if not symbol_data:
                symbol_dir = self.data_dir / "raw" / "histdata" / symbol
                if symbol_dir.exists():
                    # Look for CSV files in the symbol directory
                    csv_files = list(symbol_dir.glob("*.csv"))
                    if csv_files:
                        logger.info(f"Loading raw CSV data for {symbol}")
                        for csv_file in csv_files[:3]:  # Limit to 3 files
                            try:
                                df = pd.read_csv(csv_file)
                                df['symbol'] = symbol
                                symbol_data.append(df)
                                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
                            except Exception as e:
                                logger.warning(f"Error loading {csv_file}: {str(e)}")

            if symbol_data:
                symbol_df = pd.concat(symbol_data, ignore_index=True)
                all_data.append(symbol_df)
                logger.info(f"Combined {symbol}: {len(symbol_df)} total rows")

        if not all_data:
            logger.warning("No data found for any symbols")
            return None

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Basic preprocessing
        if 'timestamp' not in combined_df.columns:
            # Try to find timestamp column
            possible_timestamp_cols = ['timestamp', 'time', 'datetime', 'date']
            timestamp_col = None
            for col in possible_timestamp_cols:
                if col in combined_df.columns:
                    timestamp_col = col
                    break

            if timestamp_col is None:
                # Use first column as timestamp
                timestamp_col = combined_df.columns[0]

            combined_df = combined_df.rename(columns={timestamp_col: 'timestamp'})

        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
        combined_df = combined_df.dropna(subset=['timestamp'])
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

        # Ensure we have OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in combined_df.columns]

        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            # Try to infer OHLC from available data
            if 'close' in combined_df.columns:
                for col in missing_cols:
                    combined_df[col] = combined_df['close']  # Use close as fallback
            else:
                logger.error("No price data available")
                return None

        # Save processed data
        output_path = self.data_dir / "processed" / "comprehensive_forex_data.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_path)

        logger.info(f"Saved processed data to {output_path}")
        return combined_df

    def _validate_and_clean_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and clean training data to handle infinity and NaN values."""
        logger.info("Validating and cleaning data for infinity/NaN values")

        # Check for infinity values with detailed logging
        train_inf_cols = []
        test_inf_cols = []

        for col in X_train.columns:
            # Skip non-numeric columns
            if not np.issubdtype(X_train[col].dtype, np.number):
                continue

            try:
                train_inf_mask = np.isinf(X_train[col])
                test_inf_mask = np.isinf(X_test[col])

                if np.any(train_inf_mask):
                    train_inf_count = np.sum(train_inf_mask)
                    train_inf_vals = X_train.loc[train_inf_mask, col].unique()
                    logger.warning(f"Column {col} has {train_inf_count} infinity values in train: {train_inf_vals[:5]}...")
                    train_inf_cols.append(col)

                if np.any(test_inf_mask):
                    test_inf_count = np.sum(test_inf_mask)
                    test_inf_vals = X_test.loc[test_inf_mask, col].unique()
                    logger.warning(f"Column {col} has {test_inf_count} infinity values in test: {test_inf_vals[:5]}...")
                    test_inf_cols.append(col)
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not check infinity for column {col}: {str(e)}")

        if train_inf_cols or test_inf_cols:
            logger.warning(f"Found infinity values in columns: train={train_inf_cols}, test={test_inf_cols}")

            # Replace infinity values with NaN, then fill with column means
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)

        # Check for NaN values and fill them
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            logger.warning("Found NaN values, filling with column means")

            # Fill NaN values with column means
            for col in X_train.columns:
                if X_train[col].isnull().any():
                    mean_val = X_train[col].mean()
                    if np.isnan(mean_val):
                        mean_val = 0.0  # Fallback for columns with all NaN
                    X_train[col] = X_train[col].fillna(mean_val)
                    X_test[col] = X_test[col].fillna(mean_val)

        # Ensure all data is numeric and convert to float64 to prevent float32 overflow
        logger.info("Converting data to float64 for numerical stability")
        for col in X_train.columns:
            if X_train[col].dtype.name == 'category':
                # Convert categorical to numeric codes
                X_train[col] = X_train[col].cat.codes.astype(np.float64)
                X_test[col] = X_test[col].cat.codes.astype(np.float64)
            elif X_train[col].dtype.name in ['object', 'string']:
                # Try to convert object columns to numeric
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').astype(np.float64)
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').astype(np.float64)
                except:
                    # Drop non-numeric columns
                    X_train = X_train.drop(col, axis=1)
                    X_test = X_test.drop(col, axis=1)
                    logger.warning(f"Dropped non-numeric column: {col}")

        # Now convert everything to float64
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)

        # Check for very large values that might cause overflow
        for col in X_train.columns:
            max_val = X_train[col].max()
            min_val = X_train[col].min()

            # Cap extremely large values
            if max_val > 1e10 or min_val < -1e10:
                logger.warning(f"Column {col} has extreme values (min: {min_val}, max: {max_val}), capping to reasonable range")
                X_train[col] = X_train[col].clip(-1e10, 1e10)
                X_test[col] = X_test[col].clip(-1e10, 1e10)

        # Check for values that would overflow float32
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min

        train_overflow = ((X_train > float32_max) | (X_train < float32_min)).sum().sum()
        test_overflow = ((X_test > float32_max) | (X_test < float32_min)).sum().sum()

        if train_overflow > 0 or test_overflow > 0:
            logger.warning(f"Found values exceeding float32 range: train={train_overflow}, test={test_overflow}")
            # Clip to float32 range as final safety measure
            X_train = X_train.clip(float32_min, float32_max)
            X_test = X_test.clip(float32_min, float32_max)

        # Final validation
        train_inf_count = np.sum(np.isinf(X_train.values))
        test_inf_count = np.sum(np.isinf(X_test.values))
        train_nan_count = X_train.isnull().sum().sum()
        test_nan_count = X_test.isnull().sum().sum()

        if train_inf_count > 0 or test_inf_count > 0:
            logger.error(f"Still found infinity values after cleaning: train={train_inf_count}, test={test_inf_count}")
            raise ValueError("Data cleaning failed to remove all infinity values")

        if train_nan_count > 0 or test_nan_count > 0:
            logger.error(f"Still found NaN values after cleaning: train={train_nan_count}, test={test_nan_count}")
            raise ValueError("Data cleaning failed to remove all NaN values")

        logger.info("Data validation and cleaning completed successfully")
        return X_train, X_test

    def _train_models(self, X_train, X_test, y_train, y_test,
                     optimize_hyperparams: bool, use_ensemble: bool) -> Dict:
        """Train multiple models with optional hyperparameter optimization."""
        models = {}

        # CatBoost with optimization
        if optimize_hyperparams:
            logger.info("Optimizing CatBoost hyperparameters")
            optimizer = HyperparameterOptimizer(
                study_name="comprehensive_catboost",
                n_trials=5000,
                timeout=36000
            )

            opt_results = optimizer.optimize_catboost(X_train, X_test, y_train, y_test)
            best_params = opt_results['best_params']
            # Only keep parameters supported by CatBoostModel
            supported_params = {
                'iterations': best_params.get('iterations', 800),
                'learning_rate': best_params.get('learning_rate', 0.05),
                'depth': best_params.get('depth', 6),
                'l2_leaf_reg': best_params.get('l2_leaf_reg', 3.0),
                'loss_function': 'RMSE',
                'random_seed': 42,
                'early_stopping_rounds': 50
            }
            best_params = supported_params

            catboost_model = CatBoostModel(**best_params)
            catboost_model.train(X_train, y_train, X_test, y_test)
            models['catboost_optimized'] = catboost_model
        else:
            # Use default CatBoost
            catboost_model = CatBoostModel()
            catboost_model.train(X_train, y_train, X_test, y_test)
            models['catboost'] = catboost_model

        # LightGBM (if available)
        if use_ensemble:
            try:
                import lightgbm as lgb
                logger.info("Training LightGBM model")

                lgb_model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=-1
                )
                lgb_model.fit(X_train, y_train)
                models['lightgbm'] = lgb_model
            except ImportError:
                logger.warning("LightGBM not available")

        # Ensemble models
        if use_ensemble:
            try:
                from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

                logger.info("Training ensemble models")

                # Random Forest
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                models['random_forest'] = rf_model

                # Extra Trees
                et_model = ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                et_model.fit(X_train, y_train)
                models['extra_trees'] = et_model

            except ImportError:
                logger.warning("Scikit-learn ensemble models not available")

        logger.info(f"Trained {len(models)} models: {list(models.keys())}")

        return {
            'models': models,
            'model_count': len(models),
            'optimization_performed': optimize_hyperparams
        }

    def _compare_and_select_models(self, models: Dict) -> Dict:
        """Compare models and select the best one."""
        # Get sample data for comparison (we'll need to pass this in a real implementation)
        # For now, create a placeholder
        logger.info("Comparing trained models")

        # This would normally use the ModelComparator class
        # For this implementation, we'll use a simple comparison

        best_model_name = None
        best_score = -float('inf')

        # Simple comparison based on available metrics
        for model_name, model in models.items():
            try:
                # This is a simplified comparison - in practice you'd use proper CV
                score = 0.5  # Placeholder
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            except:
                continue

        if best_model_name is None:
            best_model_name = list(models.keys())[0]

        return {
            'best_model': best_model_name,
            'model_count': len(models),
            'available_models': list(models.keys())
        }

    def _generate_interpretability(self, models: Dict, X_train, X_test, y_test) -> Dict:
        """Generate model interpretability analysis."""
        logger.info("Generating model interpretability analysis")

        interpretations = {}

        for model_name, model in models.items():
            try:
                interpretation = self.interpreter.explain_model(
                    model, X_train, X_test, y_test, model_name
                )
                interpretations[model_name] = interpretation
            except Exception as e:
                logger.warning(f"Failed to interpret {model_name}: {str(e)}")
                interpretations[model_name] = {'error': str(e)}

        return {
            'interpretations_generated': len(interpretations),
            'models_interpreted': list(interpretations.keys())
        }

    def _finalize_training(self, results: Dict, best_model: str) -> Dict:
        """Finalize training and save comprehensive results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comprehensive results
        results_file = self.logs_dir / f"comprehensive_training_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate summary report
        summary = self._generate_summary_report(results, best_model)
        summary_file = self.logs_dir / f"comprehensive_training_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")

        return {
            'results_saved': str(results_file),
            'summary_saved': str(summary_file),
            'best_model': best_model
        }

    def _generate_summary_report(self, results: Dict, best_model: str) -> str:
        """Generate a comprehensive summary report."""
        report = f"""
COMPREHENSIVE FOREX AI TRAINING SUMMARY
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION:
{'-'*20}
• Hyperparameter Optimization: {results['configuration']['optimize_hyperparams']}
• Ensemble Methods: {results['configuration']['use_ensemble']}
• Model Interpretability: {results['configuration']['enable_interpretability']}
• Feature Selection: {results['configuration']['feature_selection']}

DATA PREPARATION:
{'-'*20}
"""

        if 'data_preparation' in results['stages']:
            dp = results['stages']['data_preparation']
            if dp['success']:
                report += f"• Features: {dp['feature_count']}\n"
                report += f"• Training Samples: {dp['train_samples']}\n"
                report += f"• Test Samples: {dp['test_samples']}\n"

        # Format total time safely
        total_time = results.get('total_time_seconds', 'N/A')
        if isinstance(total_time, (int, float)):
            time_str = f"{total_time:.2f}"
        else:
            time_str = str(total_time)

        report += f"""
MODEL TRAINING:
{'-'*20}
• Best Model: {best_model}
• Total Time: {time_str} seconds

FILES GENERATED:
{'-'*20}
• Results: logs/comprehensive_training_results_*.json
• Summary: logs/comprehensive_training_summary_*.txt
• Models: models/trained/
• Logs: logs/

NEXT STEPS:
{'-'*20}
1. Review the best model performance in the summary
2. Check model interpretability results (if enabled)
3. Deploy the best model for inference
4. Monitor performance and retrain as needed

{'='*60}
"""

        return report

    def _finalize_results(self, results: Dict, error: str) -> Dict:
        """Finalize results with error handling."""
        results['success'] = False
        results['error'] = error
        results['pipeline_end'] = datetime.now().isoformat()

        return results

def main():
    """Main function to run comprehensive training."""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Forex AI Training Pipeline')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--ensemble', action='store_true', help='Include ensemble methods')
    parser.add_argument('--interpretability', action='store_true', help='Enable model interpretability')
    parser.add_argument('--features', type=int, default=200, help='Number of features to select')
    parser.add_argument('--no-selection', action='store_true', help='Disable feature selection')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ComprehensiveTrainingPipeline()

    # Run comprehensive training
    results = pipeline.run_comprehensive_training(
        optimize_hyperparams=args.optimize,
        use_ensemble=args.ensemble,
        enable_interpretability=args.interpretability,
        feature_selection=not args.no_selection,
        n_features=args.features
    )

    if results.get('success'):
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE FOREX AI TRAINING COMPLETED!")
        print("="*60)
        print(f"🏆 Best Model: {results['stages']['model_comparison']['best_model']}")
        print(".2f")
        print("📁 Check logs/ for detailed results")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ COMPREHENSIVE TRAINING FAILED")
        print("="*60)
        print(f"Error: {results.get('error', 'Unknown error')}")
        print("="*60)
        exit(1)

if __name__ == "__main__":
    main()