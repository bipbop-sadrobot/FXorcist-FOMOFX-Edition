import pandas as pd
import time
import psutil
from forex_ai_dashboard.pipeline.rolling_validation import rolling_validation
from forex_ai_dashboard.utils.logger import logger
from sklearn.linear_model import LinearRegression

def run_uat():
    results = {}
    model = LinearRegression()  # Simple model for testing
    
    # Scenario 1: Basic Functionality
    try:
        data = pd.read_csv('data/uat/small_dataset.csv')
        score = rolling_validation(model, data, window_size=5, target_col='target')
        results['Scenario 1'] = 'PASS' if isinstance(score, float) else 'FAIL'
    except Exception as e:
        results['Scenario 1'] = f'FAIL: {str(e)}'
    
    # Scenario 2: Large Dataset Handling
    try:
        start_time = time.time()
        process = psutil.Process()
        start_mem = process.memory_info().rss
        
        data = pd.read_csv('data/uat/large_dataset.csv')
        rolling_validation(model, data, window_size=100)
        
        duration = time.time() - start_time
        end_mem = process.memory_info().rss
        mem_used = (end_mem - start_mem) / (1024 ** 2)  # MB
        
        passed = duration < 120 and mem_used < 1000
        results['Scenario 2'] = 'PASS' if passed else f'FAIL (Time: {duration:.1f}s, Mem: {mem_used:.1f}MB)'
    except Exception as e:
        results['Scenario 2'] = f'FAIL: {str(e)}'
    
    # Scenario 3: Custom Metric
    try:
        def custom_mae(y_true, y_pred):
            return abs(y_true - y_pred).mean()
            
        data = pd.read_csv('data/uat/small_dataset.csv')
        score = rolling_validation(model, data, window_size=5, metric=custom_mae)
        results['Scenario 3'] = 'PASS' if isinstance(score, float) else 'FAIL'
    except Exception as e:
        results['Scenario 3'] = f'FAIL: {str(e)}'
    
    # Scenario 4: Error Handling
    try:
        data = pd.read_csv('data/uat/invalid_dataset.csv')
        rolling_validation(None, data, window_size=3)
        results['Scenario 4'] = 'FAIL: Did not raise error'
    except ValueError as e:
        results['Scenario 4'] = 'PASS'
    except Exception as e:
        results['Scenario 4'] = f'FAIL: Unexpected error ({str(e)})'
    finally:
        pass
    
    # Print results
    print("\nUAT Results:")
    for scenario, result in results.items():
        print(f"{scenario}: {result}")

if __name__ == '__main__':
    run_uat()
