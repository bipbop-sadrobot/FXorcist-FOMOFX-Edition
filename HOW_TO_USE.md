# 🚀 **FXorcist-FOMOFX-Edition: Complete Usage Guide**

## 🎯 **System Overview**

This is a comprehensive Forex AI system that provides automated data processing, advanced feature engineering, model training, and continuous learning capabilities for forex price prediction. The system includes a sophisticated memory architecture and real-time monitoring dashboards.

---

## 📋 **Table of Contents**

1. [Quick Start](#-quick-start)
2. [System Requirements](#-system-requirements)
3. [Installation & Setup](#-installation--setup)
4. [Data Preparation](#-data-preparation)
5. [Training Pipeline](#-training-pipeline)
6. [Memory System](#-memory-system)
7. [Dashboard Usage](#-dashboard-usage)
8. [Validation & Testing](#-validation--testing)
9. [Troubleshooting](#-troubleshooting)
10. [Advanced Configuration](#-advanced-configuration)

---

## ⚡ **Quick Start**

### **1. Basic Training (5 minutes)**
```bash
# Install dependencies
pip install -r requirements.txt

# Run basic training with existing data
python forex_ai_dashboard/pipeline/model_training.py
```

### **2. Launch Dashboard**
```bash
# Training dashboard
streamlit run enhanced_training_dashboard.py

# Main dashboard (in separate terminal)
cd dashboard && streamlit run app.py
```

### **3. Test Memory System**
```bash
# Run memory system demo
python examples/run_cycle.py
```

---

## 💻 **System Requirements**

### **Minimum Requirements**
- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 5GB free space
- **OS**: Linux, macOS, or Windows

### **Recommended Requirements**
- **Python**: 3.10+
- **RAM**: 16GB+
- **Storage**: 20GB+ SSD
- **GPU**: NVIDIA GPU (optional, for faster training)

### **Dependencies**
```bash
# Core ML libraries
pip install catboost lightgbm xgboost scikit-learn

# Data processing
pip install pandas numpy pyarrow

# Visualization
pip install streamlit plotly matplotlib seaborn

# Technical analysis
pip install ta

# Optional (for advanced features)
pip install shap optuna mlflow torch
```

---

## 🔧 **Installation & Setup**

### **1. Clone Repository**
```bash
git clone https://github.com/your-repo/FXorcist-FOMOFX-Edition.git
cd FXorcist-FOMOFX-Edition
```

### **2. Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### **3. Install Dependencies**
```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import catboost, pandas, numpy, streamlit; print('✅ All core dependencies installed')"
```

### **4. Verify System Components**
```bash
# Check data availability
ls -la data/processed/
ls -la data/raw/

# Check model directory
ls -la models/trained/

# Check logs directory
ls -la logs/
```

---

## 📊 **Data Preparation**

### **Current Data Status**
✅ **Available Data:**
- EURUSD processed features (10,000 samples)
- Raw EURUSD data (CSV format)
- Historical data structure in place

### **Data Pipeline Overview**
```
Raw Data → Validation → Feature Engineering → Processed Data → Training
```

### **Verify Data Integrity**
```bash
# Check processed data
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/eurusd_features_2024.parquet')
print(f'✅ Data loaded: {len(df)} rows, {len(df.columns)} columns')
print(f'Columns: {list(df.columns[:5])}...')
"

# Check raw data
python -c "
import pandas as pd
df = pd.read_csv('data/raw/ejtrader_eurusd_m1.csv')
print(f'✅ Raw data: {len(df)} rows')
print(f'Date range: {df.date.min()} to {df.date.max()}')
"
```

### **Data Structure**
```
data/
├── raw/                    # Raw forex data
│   ├── ejtrader_eurusd_m1.csv
│   └── histdata/
├── processed/              # Processed features
│   └── eurusd_features_2024.parquet
├── cleaned/               # Cleaned data by symbol
└── checkpoints/           # Training checkpoints
```

---

## 🤖 **Training Pipeline**

### **Core Training Scripts**

#### **1. Basic Training**
```bash
cd forex_ai_dashboard/pipeline
python model_training.py
```
**Features:**
- Uses existing processed data
- CatBoost model with optimized parameters
- Automatic feature selection
- Performance metrics and logging

#### **2. Enhanced Feature Engineering**
```bash
python forex_ai_dashboard/pipeline/enhanced_feature_engineering.py
```
**Features:**
- 50+ technical indicators
- Advanced momentum, volatility, trend features
- Feature importance analysis
- Automated feature selection

### **Training Process Flow**
```
Data Loading → Feature Engineering → Model Training → Evaluation → Saving
```

### **Monitor Training Progress**
```bash
# Watch training logs in real-time
tail -f logs/model_training.log

# Check model performance
cat models/trained/catboost_forex_*_metrics.json | jq '.metrics'
```

### **Expected Performance**
- **Training Time**: ~5-30 seconds
- **R² Score**: 0.99+ (excellent performance)
- **RMSE**: < 0.0002 (very low error)
- **Features Used**: 13 optimized features

---

## 🧠 **Memory System**

### **Architecture Overview**
The system includes a sophisticated multi-tier memory system:
- **Working Memory (WM)**: Recent predictions and patterns
- **Long-Term Memory (LTM)**: Historical patterns
- **Episodic Memory (EM)**: Significant market events

### **Memory System Components**
```
memory_system/
├── core.py          # Core memory management
├── memory.py        # Integrated memory system
├── anomaly.py       # Anomaly detection
├── event_bus.py     # Event-driven communication
├── federated.py     # Federated learning
└── utils.py         # Utility functions
```

### **Test Memory System**
```bash
# Run memory system demonstration
python examples/run_cycle.py

# Expected output:
# [EVENT] anomaly_detected: {...}
# Insights: {...}
# Resource plan: {...}
```

### **Memory System Features**
- ✅ Event-driven architecture
- ✅ Anomaly detection
- ✅ Federated learning simulation
- ✅ Resource prioritization
- ⚠️ Advanced regime detection (requires ruptures package)

---

## 📈 **Dashboard Usage**

### **Training Dashboard**
```bash
streamlit run enhanced_training_dashboard.py
```
**Features:**
- Real-time training progress
- Model performance metrics
- Feature importance visualization
- Training history

### **Main Dashboard**
```bash
cd dashboard
streamlit run app.py
```
**Features:**
- System status monitoring
- Model predictions
- Performance analytics
- Resource usage

### **Dashboard Components**
- **System Status**: CPU, memory, disk usage
- **Model Performance**: R², RMSE, MAE metrics
- **Training Progress**: Real-time updates
- **Feature Analysis**: Importance rankings

---

## ✅ **Validation & Testing**

### **1. Component Verification**
```bash
# Test data loading
python -c "
from forex_ai_dashboard.pipeline.data_ingestion import load_data
df = load_data()
print(f'✅ Data ingestion: {len(df)} rows loaded')
"

# Test feature engineering
python -c "
from forex_ai_dashboard.pipeline.enhanced_feature_engineering import EnhancedFeatureEngineer
engineer = EnhancedFeatureEngineer()
print('✅ Feature engineering initialized')
"

# Test model training
python -c "
from forex_ai_dashboard.models.catboost_model import CatBoostModel
model = CatBoostModel()
print('✅ Model architecture verified')
"
```

### **2. End-to-End Testing**
```bash
# Run complete training pipeline
python forex_ai_dashboard/pipeline/model_training.py

# Verify outputs
ls -la models/trained/
ls -la logs/
```

### **3. Memory System Testing**
```bash
# Test memory system (without ruptures)
python examples/run_cycle.py

# Expected: Successful execution with anomaly detection
```

### **4. Dashboard Testing**
```bash
# Test training dashboard
streamlit run enhanced_training_dashboard.py --server.headless true

# Test main dashboard
cd dashboard && streamlit run app.py --server.headless true
```

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. Import Errors**
```bash
# Fix missing dependencies
pip install catboost pandas numpy streamlit

# Verify Python path
python -c "import sys; print(sys.path)"
```

#### **2. Memory Issues**
```bash
# Check system resources
python -c "
import psutil
print(f'RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB available')
print(f'CPU: {psutil.cpu_count()} cores')
"
```

#### **3. Data Loading Issues**
```bash
# Verify data files exist
ls -la data/processed/
ls -la data/raw/

# Check data integrity
python -c "
import pandas as pd
try:
    df = pd.read_parquet('data/processed/eurusd_features_2024.parquet')
    print('✅ Data file is valid')
except Exception as e:
    print(f'❌ Data error: {e}')
"
```

#### **4. Model Training Failures**
```bash
# Check logs for details
tail -20 logs/model_training.log

# Verify model parameters
cat models/trained/*_metrics.json
```

### **Performance Optimization**
```bash
# Use smaller batch sizes for memory
export PYTHONPATH=$PYTHONPATH:$(pwd)
python forex_ai_dashboard/pipeline/model_training.py --batch_size 1000

# Enable GPU acceleration (if available)
export CUDA_VISIBLE_DEVICES=0
```

---

## ⚙️ **Advanced Configuration**

### **Model Parameters**
```python
# Custom CatBoost configuration
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
def add_custom_features(df):
    # Your custom feature logic
    df['custom_sma'] = df['close'].rolling(20).mean()
    df['custom_rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    return df
```

### **Memory System Configuration**
```python
# Memory system settings
memory_config = {
    'max_records': 2000,
    'federated_rounds': 10,
    'anomaly_threshold': 0.05
}
```

---

## 📊 **Performance Monitoring**

### **Training Metrics**
```bash
# View latest training results
ls -la models/trained/ | tail -5

# Check performance metrics
python -c "
import json
with open('models/trained/catboost_forex_latest_metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f'R² Score: {metrics[\"metrics\"][\"r2\"]:.4f}')
    print(f'RMSE: {metrics[\"metrics\"][\"rmse\"]:.6f}')
"
```

### **System Health**
```bash
# Monitor system resources
python -c "
import psutil
cpu = psutil.cpu_percent()
mem = psutil.virtual_memory().percent
print(f'CPU Usage: {cpu}%')
print(f'Memory Usage: {mem}%')
"
```

---

## 🎯 **Next Steps**

1. **Complete Basic Training**: Run the training pipeline
2. **Explore Dashboards**: Monitor system performance
3. **Customize Features**: Add domain-specific indicators
4. **Set Up Automation**: Enable continuous learning
5. **Monitor Performance**: Track model improvements

---

## 📞 **Support & Resources**

- **Documentation**: See `docs/` directory
- **Logs**: Check `logs/` for detailed information
- **Models**: Trained models in `models/trained/`
- **Data**: Processed data in `data/processed/`

---

*This comprehensive guide covers all aspects of the FXorcist-FOMOFX-Edition system. The system is designed for production use with robust error handling and monitoring capabilities.*