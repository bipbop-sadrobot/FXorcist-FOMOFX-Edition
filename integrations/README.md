# FXorcist Integration Bundle

This integration bundle adds advanced analytics, backtesting, and ML capabilities to FXorcist through:

- QuantStats for rich analytics & reporting
- VectorBT for fast backtesting
- Alphalens for signal/factor analysis
- MlFinLab for advanced labeling & CV
- EconML for causal analysis
- Model Zoo improvements

## Quick Start

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r integrations/requirements.txt
```

Note: If VectorBT installation fails on macOS, use Conda:
```bash
conda create -n fx python=3.11
conda activate fx
pip install -r integrations/requirements.txt
```

## Usage Examples

### Generate QuantStats Report
```bash
python integrations/reports/quantstats_report.py \
  --returns_csv path/to/returns.csv \
  --returns_col returns \
  --out integrations/artifacts/qstats_report.html
```

### Run VectorBT Parameter Sweep
```bash
python integrations/backtest/vectorbt_runner.py \
  --csv path/to/prices.csv \
  --fast "5,10,20" \
  --slow "50,100,200" \
  --out integrations/artifacts/vectorbt_sweep.csv
```

### Analyze Signal with Alphalens
```bash
python integrations/reports/alphalens_report.py \
  --factor_csv path/to/factor.csv \
  --price_csv path/to/prices.csv \
  --out integrations/artifacts/alphalens_summary.csv
```

### Launch Dashboard
```bash
streamlit run integrations/ui/streamlit_app.py
```

## Integration Points

### Consolidation Worker
- Generate QuantStats reports after backtests
- Run Alphalens analysis on predictions
- Execute decomposition tests
- Update model artifacts

### Model Training
- Use MlFinLab for robust labeling
- Implement purged CV
- Auto-evaluate model candidates

### Real-time Processing
- Update causal effects estimates
- Apply trade gating rules
- Monitor performance metrics

## Directory Structure

```
integrations/
├── README.md
├── requirements.txt
├── INTEGRATION_PLAN.md
├── reports/
│   ├── quantstats_report.py
│   └── alphalens_report.py
├── backtest/
│   └── vectorbt_runner.py
├── features/
│   └── mlfinlab_tools.py
├── ui/
│   └── streamlit_app.py
├── fxorcist_integration/
│   ├── causal/
│   │   └── econml_effects.py
│   ├── tests/
│   │   └── decomposed_tests.py
│   └── models/
│       └── model_zoo.py
└── artifacts/
    └── .gitkeep
```

## Important Notes

1. EconML requires careful validation of causal assumptions
2. MlFinLab has some premium features (using open-source subset)
3. VectorBT works best with Conda on macOS
4. Consider regime changes in Forex when training models

See INTEGRATION_PLAN.md for detailed implementation timeline and technical notes.