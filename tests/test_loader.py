import pytest
import pandas as pd
from fxorcist.data.loader import load_symbol, _synthetic_series, list_available_symbols
from pathlib import Path

def test_synthetic_series():
    df = _synthetic_series('TST', n=10, seed=1)
    assert len(df) == 10
    assert 'Close' in df.columns

def test_load_symbol_fallback(tmp_path):
    with pytest.raises(Exception):
        load_symbol('NOFILE', base_dir=str(tmp_path))
    df = load_symbol('NOFILE', base_dir=str(tmp_path), allow_synthetic_fallback=True)
    assert not df.empty

def test_list_available_symbols(tmp_path):
    (tmp_path / 'data' / 'cleaned').mkdir(parents=True)
    p = tmp_path / 'data' / 'cleaned' / 'E1.csv'
    p.write_text('Date,Open,High,Low,Close\n2020-01-01,1,1,1,1\n')
    syms = list_available_symbols(base_dir=str(tmp_path))
    assert 'E1' in syms
