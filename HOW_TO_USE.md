# How to Use the Forex AI Dashboard

[Existing content remains unchanged]

## Number Analysis Script

The `number_analysis_improved.py` script is used to analyze a list of numbers and identify patterns. It can be used to score numbers based on the rarity of patterns. The script takes a CSV, JSON, or TXT file as input and saves the results to `analysis_results.csv`.

To run the script, use the following command:

```bash
python number_analysis_improved.py
```

## Development Mode Setup

For local development with live data:

1. Create `.env` file with API keys
2. Start development server:
   ```bash
   python dashboard/app.py --env=development --data_path=./data
   ```
3. Access debugging dashboard at `http://localhost:8050/dev`

## User Acceptance Testing (UAT)

1. Install required dependencies:
```bash
pip install -r requirements.txt
pip install psutil  # For memory monitoring
```

2. Prepare test datasets:
```bash
mkdir -p data/uat
python -c "import pandas as pd; pd.DataFrame({'feature1':[1,2],'feature2':[3,4],'target':[5,6]}).to_csv('data/uat/small_dataset.csv', index=False)"
python -c "import pandas as pd, numpy as np; pd.DataFrame(np.random.rand(10000,3), columns=['feature1','feature2','target']).to_csv('data/uat/large_dataset.csv', index=False)"
python -c "import pandas as pd; pd.DataFrame({'feature1':[1,2],'feature2':[3,4]}).to_csv('data/uat/invalid_dataset.csv', index=False)"
```

3. Run the UAT test script:
```bash
python uat/run_rolling_validation_uat.py
```

4. Review the test results printed in the console:
```
UAT Results:
Scenario 1: Basic Functionality - PASS
Scenario 2: Large Dataset Handling - PASS
Scenario 3: Custom Metric - PASS
Scenario 4: Error Handling - PASS
```

5. Report any failures through:
- GitHub Issues: https://github.com/yourrepo/issues
- Email: uat@yourcompany.com

For full UAT documentation see [UAT_ROLLING_VALIDATION.md](UAT_ROLLING_VALIDATION.md)
