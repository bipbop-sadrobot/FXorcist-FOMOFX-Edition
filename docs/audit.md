# FXorcist-FOMOFX-Edition — Refactor Audit

> Generated: 2025-09-08  
> Repo: FXorcist-FOMOFX-Edition

## Top 10 Refactor Targets

1. **`backtest.py`** — Monolithic. Extract:
   - Event replay engine
   - Execution simulation
   - Portfolio management
   - Performance metrics
   - Risk analytics

2. **`strategy.py`** — Hardcoded logic. Extract:
   - Base strategy class
   - Technical indicators
   - Signal generation
   - Position sizing
   - Risk management

3. **`data_loader.py`** — No abstraction. Extract:
   - Data source connectors (CSV, exchange, DB)
   - Data cleaning pipeline
   - Feature engineering
   - Market data caching
   - Real-time feed handling

4. **`config.py` (missing)** — Add:
   - YAML config loader
   - Environment variables
   - Schema validation
   - Defaults management
   - Config versioning

5. **No CLI** — Add `fxorcist/cli.py`:
   - Modern Typer interface
   - Rich progress bars
   - Subcommands (prepare, backtest, optimize)
   - Config validation
   - Error handling

6. **No package structure** — Move code under `fxorcist/`:
   - Proper Python package
   - Module organization
   - Import management
   - Version handling
   - Package metadata

7. **No tests** — Add `tests/`:
   - Unit tests (pytest)
   - Integration tests
   - Performance benchmarks
   - Coverage reporting
   - CI pipeline

8. **Global state** — Portfolio/state mutated globally:
   - Use immutable snapshots
   - Event sourcing pattern
   - State transitions
   - Audit trail
   - Replay capability

9. **Look-ahead risk** — No timestamp filtering:
   - Event bus with replay
   - Access control
   - Time-series validation
   - Data point verification
   - Bias prevention

10. **No CI/CD** — Add `.github/workflows/ci.yml`:
    - Automated testing
    - Code quality checks
    - Documentation builds
    - Release automation
    - Deployment pipeline

## Duplicate / Entangled Modules

- `backtest.py` contains strategy, execution, and metric logic → split into `backtest/`
- `data_loader.py` and `csv_reader.py` → merge into `data/connectors/`
- Multiple hardcoded commission/slippage formulas → extract to `backtest/slippage.py`
- Scattered logging calls → unified logging system
- Duplicate market data handling → centralized market data service
- Mixed business/technical logic → proper separation of concerns

## Target Architecture Layout

```
fxorcist/
├── __init__.py
├── cli.py                 # Typer CLI interface
├── config.py             # Configuration management
├── backtest/
│   ├── __init__.py
│   ├── engine.py         # Event-driven core
│   ├── portfolio.py      # Portfolio management
│   ├── execution.py      # Order execution
│   ├── risk.py          # Risk management
│   └── metrics.py        # Performance metrics
├── data/
│   ├── __init__.py
│   ├── loader.py         # Data loading interface
│   └── connectors/       # Data source implementations
├── strategies/
│   ├── __init__.py
│   ├── base.py          # Strategy base class
│   └── indicators/      # Technical indicators
└── utils/
    ├── __init__.py
    ├── logging.py       # Centralized logging
    └── validation.py    # Data validation
```

## Implementation Priority

1. Event-driven backtest engine (core functionality)
2. CLI interface (user experience)
3. Data abstraction layer (flexibility)
4. Testing infrastructure (reliability)
5. Configuration management (maintainability)
6. Package structure (organization)
7. CI/CD pipeline (automation)

## Migration Strategy

1. Create new package structure
2. Move existing code with minimal changes
3. Add tests for current functionality
4. Refactor one module at a time
5. Validate with integration tests
6. Document new architecture
7. Release and deprecate old code

## Acceptance Criteria

- [ ] All code moved to proper package structure
- [ ] Event-driven backtest engine implemented
- [ ] CLI interface functional
- [ ] Tests passing with >80% coverage
- [ ] CI pipeline operational
- [ ] Documentation updated
- [ ] No global state
- [ ] Look-ahead bias prevented
- [ ] Performance metrics validated