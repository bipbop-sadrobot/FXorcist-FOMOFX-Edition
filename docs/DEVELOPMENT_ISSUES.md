# Active Development Issues

## [[HIGH] Model Tracking Latency](https://github.com/yourrepo/issues/45)
**Module:** reinforcement/model_tracker.py  
**Last Updated:** 2025-08-14  

### Current Behavior
Model predictions experience 350-500ms latency when tracking is enabled

### Proposed Solution
- Implement asynchronous logging
- Use memory-mapped files for shared state
- Add batch processing for tracking events

---

## [[MEDIUM] Feature Versioning Conflicts](https://github.com/yourrepo/issues/46)
**Module:** pipeline/feature_engineering.py  
**Last Updated:** 2025-08-13  

### Description
Feature mismatches occur when rolling back model versions

### Proposed Fix
- Add feature schema versioning
- Implement automatic feature compatibility checks

---

## [[LOW] Dashboard Timezone Handling](https://github.com/yourrepo/issues/47)
**Module:** dashboard/app.py  
**Last Updated:** 2025-08-12  

### Issue
UTC conversion inconsistent across visualizations

### Solution
- Centralize timezone management
- Add timezone selector in UI
