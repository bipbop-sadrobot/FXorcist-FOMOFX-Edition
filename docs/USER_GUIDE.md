# FXorcist User Guide

## Overview
This guide covers how to run the FXorcist forex AI pipeline, train models, and use the dashboard.

## Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the dashboard:
```bash
cd dashboard && streamlit run app.py
```

## Pipeline Operations

### Running the Pipeline
1. Data Ingestion:
```bash
python scripts/fetch_data.sh  # Fetch latest forex data
python data/ingestion.py      # Process raw data
```

2. Data Validation:
- Automatic checks for data integrity
- Quarantine system for suspicious data
- View quarantined entries in dashboard

3. Feature Engineering:
- Technical indicators
- Market regime detection
- Memory-based features

4. Model Training:
```bash
python forex_ai_dashboard/pipeline/model_training.py
```

### Memory System

The system maintains an intelligent memory of market patterns and predictions:

1. Memory Tiers:
- Working Memory (WM): Recent patterns
- Long-Term Memory (LTM): Historical insights
- Episodic Memory (EM): Significant events

2. Memory Dashboard:
- View memory usage stats
- Monitor recall latency
- Check quarantined entries
- View memory insights

### Dashboard Components

1. System Status:
- Resource monitoring
- Memory system metrics
- Pipeline health checks

2. Predictions:
- Real-time forecasts
- Confidence metrics
- Historical accuracy

3. Performance:
- Model evaluation metrics
- Memory system performance
- System resource usage

## Troubleshooting

### Common Issues

1. Memory System:
- High recall latency: Check system resources
- Missing predictions: Verify data pipeline
- Quarantined entries: Review data quality

2. Pipeline:
- Data ingestion errors: Check source availability
- Training failures: Review logs
- Dashboard connection: Verify services running

### Health Checks

1. System Status:
- Green: All systems operational
- Yellow: Performance degradation
- Red: Critical issues

2. Memory Health:
- Usage trends
- Quarantine ratio
- Recall performance

## Best Practices

1. Regular Maintenance:
- Monitor memory usage
- Review quarantined data
- Clean old entries

2. Performance Optimization:
- Use batch operations for bulk data
- Monitor recall latency
- Keep memory tiers balanced

3. Data Quality:
- Review quarantined entries
- Validate new data sources
- Monitor drift metrics

## Support

For technical issues:
1. Check logs in `logs/`
2. Review system status dashboard
3. Consult development documentation