# 📚 **Forex AI System - Complete Usage Guide**

## 🎯 **System Overview**

This Forex AI system provides automated data downloading, processing, model training, and continuous learning capabilities for forex price prediction.

---

## 📁 **File Structure & Usage**

### 🔧 **Core Training Scripts**

#### 1. `simple_train.py` - Quick Start Training
**Purpose**: Fast, simple training with existing data
**Usage**:
```bash
python simple_train.py
```
**Features**:
- Uses existing processed data (`data/processed/eurusd_features_2024.parquet`)
- Trains CatBoost model with basic features
- Saves model to `models/trained/simple_catboost_*.cbm`
- Generates performance metrics

**Output**:
- Trained model file
- Performance metrics (RMSE, MAE, R²)
- Training logs

#### 2. `focused_training_pipeline.py` - Advanced Training
**Purpose**: Comprehensive training with enhanced features
**Usage**:
```bash
python focused_training_pipeline.py
```
**Features**:
- Downloads recent data (2023-2024) for EURUSD, GBPUSD
- Advanced feature engineering (25+ indicators)
- Optimized CatBoost with forex-specific parameters
- Comprehensive logging and metrics

**Output**:
- Enhanced processed data (`data/processed/focused_forex_data.parquet`)
- Optimized model (`models/trained/catboost_optimized_*.cbm`)
- Feature importance analysis
- Detailed training summary

#### 3. `automated_training_pipeline.py` - Full Automation
**Purpose**: Complete end-to-end automation
**Usage**:
```bash
python automated_training_pipeline.py
```
**Features**:
- Downloads data for 10 major pairs (2020-2024)
- Processes and combines all data
- Trains multiple models (CatBoost, XGBoost)
- Comprehensive error handling and logging

**Output**:
- Combined dataset (`data/processed/comprehensive_forex_data.parquet`)
- Multiple trained models
- Training summary JSON
- Performance comparison

#### 4. `continuous_training_scheduler.py` - Production Automation
**Purpose**: Continuous training and monitoring
**Usage**:
```bash
python continuous_training_scheduler.py
```
**Features**:
- Scheduled training (daily/weekly/monthly)
- Performance monitoring and alerts
- Automatic model cleanup
- Daily performance reports

**Configuration**:
- Daily training: 02:00
- Weekly comprehensive: Every 7 days
- Monthly full retrain: Every 30 days
- Performance monitoring: Every 6 hours

---

### 📊 **Data Processing Scripts**

#### 5. `scripts/fetch_data.sh` - Data Downloader
**Purpose**: Download forex data from histdata.com
**Usage**:
```bash
# Download specific symbol/year/month
bash scripts/fetch_data.sh --symbols EURUSD --year 2024 --month 08

# Download with different source
bash scripts/fetch_data.sh --source histdata --symbols GBPUSD --year 2024 --month 09
```
**Parameters**:
- `--symbols`: Forex pair (e.g., EURUSD, GBPUSD)
- `--year`: Year (2020-2024)
- `--month`: Month (01-12)
- `--source`: Data source (default: histdata)

**Output**: CSV files in `data/raw/histdata/{SYMBOL}/{YEAR}/{MONTH}.csv`

#### 6. `scripts/run_pipeline.sh` - Batch Processing
**Purpose**: Process multiple symbols and months
**Usage**:
```bash
# Process EURUSD and GBPUSD for 2024 Q3
bash scripts/run_pipeline.sh 2024 07 EURUSD,GBPUSD

# Process with default parameters
bash scripts/run_pipeline.sh
```
**Parameters**:
- `YEAR`: Year (default: 2024)
- `MONTH`: Starting month (default: 07)
- `SYMBOLS`: Comma-separated pairs (default: EURUSD,GBPUSD)

**Output**:
- Cleaned data: `data/cleaned/{SYMBOL}/{YEAR}_{MONTH}.parquet`
- Processed data: `data/processed/`

#### 7. `scripts/clean_data.py` - Data Cleaning
**Purpose**: Clean and validate raw forex data
**Usage**:
```bash
python scripts/clean_data.py --input data/raw/histdata/EURUSD/2024/08.csv --output data/cleaned/EURUSD/2024_08.parquet
```
**Parameters**:
- `--input`: Input CSV file
- `--output`: Output parquet file

#### 8. `scripts/prep_data.py` - Data Preparation
**Purpose**: Prepare data for training with train/val/test splits
**Usage**:
```bash
python scripts/prep_data.py --input_dir data/cleaned/2024 --output_dir data/processed --train_pct 0.7 --val_pct 0.15
```
**Parameters**:
- `--input_dir`: Directory with cleaned data
- `--output_dir`: Output directory
- `--train_pct`: Training data percentage (default: 0.7)
- `--val_pct`: Validation data percentage (default: 0.15)

---

### 🤖 **Model Training Scripts**

#### 9. `forex_ai_dashboard/pipeline/model_training.py` - Dashboard Training
**Purpose**: Training script integrated with dashboard
**Usage**:
```bash
cd forex_ai_dashboard/pipeline
python model_training.py
```
**Features**:
- Dashboard-compatible training
- Real-time progress updates
- Model evaluation metrics
- Integration with existing dashboard

#### 10. `data/ingestion.py` - Data Ingestion Pipeline
**Purpose**: Process and validate multiple data sources
**Usage**:
```bash
python data/ingestion.py
```
**Features**:
- Parallel processing of multiple sources
- Data validation and cleaning
- Checkpoint saving
- Comprehensive logging

**Supported Sources**:
- histdata (ZIP/CSV files)
- dukascopy (CSV format)
- ejtrader (CSV format)
- fx1min (CSV format)
- algo_duka (CSV format)

---

### 📈 **Model Architecture Files**

#### 11. `forex_ai_dashboard/models/catboost_model.py` - CatBoost Model
**Purpose**: CatBoost implementation for forex prediction
**Usage**:
```python
from forex_ai_dashboard.models.catboost_model import CatBoostModel

model = CatBoostModel(
    iterations=1000,
    learning_rate=0.05,
    depth=6
)
model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

#### 12. `forex_ai_dashboard/models/lstm_model.py` - LSTM Model
**Purpose**: Deep learning model for sequence prediction
**Features**:
- Time series forecasting
- Sequence processing
- PyTorch implementation

#### 13. `forex_ai_dashboard/models/tft_model.py` - Temporal Fusion Transformer
**Purpose**: Advanced transformer-based forecasting
**Features**:
- Multi-horizon forecasting
- Attention mechanisms
- Complex feature interactions

#### 14. `forex_ai_dashboard/models/model_hierarchy.py` - Hierarchical Training
**Purpose**: Multi-layer model system
**Usage**:
```python
from forex_ai_dashboard.models.model_hierarchy import ModelHierarchy

hierarchy = ModelHierarchy()
await hierarchy.initialize_models(feature_lists)
await hierarchy.train_hierarchy(data, targets)
```

---

### 🎛️ **Dashboard & UI**

#### 15. `dashboard/app.py` - Main Dashboard
**Purpose**: Streamlit dashboard for model monitoring
**Usage**:
```bash
cd dashboard
streamlit run app.py
```
**Features**:
- Real-time model performance
- Training progress visualization
- Prediction results
- Model comparison

#### 16. `dashboard/utils/data_loader.py` - Data Loading Utilities
**Purpose**: Load and preprocess data for dashboard
**Features**:
- Efficient data loading
- Real-time data updates
- Caching mechanisms

#### 17. `dashboard/components/performance.py` - Performance Components
**Purpose**: Dashboard components for performance visualization
**Features**:
- Metrics visualization
- Performance charts
- Model comparison tools

---

### 🔧 **Utility Scripts**

#### 18. `examples/run_cycle.py` - Memory System Demo
**Purpose**: Demonstrate memory system functionality
**Usage**:
```bash
python examples/run_cycle.py
```
**Features**:
- Federated learning simulation
- Memory system demonstration
- Anomaly detection

#### 19. `memory_system/` - Memory Management
**Purpose**: Advanced memory and learning system
**Components**:
- `memory.py`: Core memory management
- `federated.py`: Federated learning
- `anomaly.py`: Anomaly detection
- `event_bus.py`: Event-driven architecture

---

## 🚀 **Quick Start Guide**

### 1. **Basic Training (5 minutes)**
```bash
# Quick training with existing data
python simple_train.py
```

### 2. **Advanced Training (15 minutes)**
```bash
# Download and train with enhanced features
python focused_training_pipeline.py
```

### 3. **Full Automation (30+ minutes)**
```bash
# Complete end-to-end pipeline
python automated_training_pipeline.py
```

### 4. **Production Setup**
```bash
# Continuous training and monitoring
python continuous_training_scheduler.py
```

---

## 📊 **Data Download Status**

### ✅ **Working Downloads**
- **EURUSD 2024**: Full year available
- **GBPUSD 2024**: Full year available
- **Recent data (2023-2024)**: Generally available

### ❌ **Failed Downloads**
- **2020-2022 data**: Not available from histdata.com
- **Some 2023 data**: May be rate-limited
- **Exotic pairs**: Limited availability

### 🔧 **Download Troubleshooting**

1. **Rate Limiting**: Add delays between requests
```bash
# In scripts, increase sleep time
await asyncio.sleep(2.0)  # Increase from 0.5
```

2. **Data Availability**: Check histdata.com directly
```bash
# Manual download check
curl -s "https://www.histdata.com/download-free-forex-data/?/ascii/1-minute-bar-quotes/EURUSD/2024/08"
```

3. **Alternative Sources**: Consider other data providers
- Dukascopy
- FXCM
- Local data files

---

## 📈 **Performance Expectations**

### **Simple Training**
- **Data**: ~10K samples
- **Features**: 13 basic features
- **Training Time**: ~5 seconds
- **Expected R²**: 0.99+

### **Focused Training**
- **Data**: ~50K+ samples
- **Features**: 25+ advanced features
- **Training Time**: ~2-5 minutes
- **Expected R²**: 0.995+

### **Automated Training**
- **Data**: ~100K+ samples (if available)
- **Features**: 25+ features
- **Training Time**: ~10-30 minutes
- **Expected R²**: 0.997+

---

## 🔧 **Configuration & Customization**

### **Model Parameters**
```python
# CatBoost configuration
model_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 1.0,
    'early_stopping_rounds': 100
}
```

### **Feature Engineering**
```python
# Add custom indicators
def add_custom_indicators(df):
    # Your custom feature logic
    df['custom_indicator'] = df['close'] * df['volume']
    return df
```

### **Training Schedule**
```python
# Customize training frequency
schedule_config = {
    'daily_training': '02:00',
    'weekly_training': True,
    'monthly_retraining': True,
    'performance_monitoring': '*/6'  # Every 6 hours
}
```

---

## 📋 **Monitoring & Maintenance**

### **Log Files**
- `logs/simple_training.log`
- `logs/focused_training.log`
- `logs/automated_training.log`
- `logs/continuous_training.log`

### **Model Storage**
- `models/trained/` - Trained models
- `models/trained/*_metrics.json` - Performance metrics
- `data/checkpoints/` - Training checkpoints

### **Performance Monitoring**
```bash
# Check recent training results
tail -f logs/continuous_training.log

# View model performance
cat models/trained/*_metrics.json | jq '.metrics'
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Data Download Failures**
   - Check internet connection
   - Verify histdata.com availability
   - Use focused pipeline for reliable data

2. **Memory Issues**
   - Reduce batch sizes in training
   - Use focused pipeline for smaller datasets
   - Monitor system resources

3. **Model Performance**
   - Check data quality
   - Verify feature engineering
   - Adjust model parameters

### **Debug Commands**
```bash
# Check data availability
ls -la data/raw/histdata/

# Verify processed data
python -c "import pandas as pd; print(pd.read_parquet('data/processed/eurusd_features_2024.parquet').shape)"

# Test model loading
python -c "from catboost import CatBoostRegressor; model = CatBoostRegressor(); model.load_model('models/trained/simple_catboost_*.cbm')"
```

---

## 🎯 **Next Steps**

1. **Run Simple Training**: Get started quickly
2. **Explore Dashboard**: Monitor training progress
3. **Customize Features**: Add domain-specific indicators
4. **Set Up Automation**: Enable continuous learning
5. **Monitor Performance**: Track model improvements

---

*This guide covers all major components of the Forex AI system. Each script includes comprehensive logging and error handling for production use.*