"""
Feature Engineering Pipeline for OHLCV-like Time Series
-------------------------------------------------------
A production-ready, extensible module to generate a rich set of technical and
statistical features from price data. Includes batching utilities, optional
externals merge, event features, Fourier & Hilbert terms, wavelet denoising,
fractional differencing (auto), and a lightweight Lorentzian KDE classifier.

Usage
-----
>>> feats = engineer_features(df, events=ev_df, externals=externals)

Requirements (core): pandas, numpy, scipy, statsmodels
Optional: pywt (wavelets), scipy.signal.hilbert (Hilbert transform)

Notes
-----
- This module expects a DataFrame with columns: ['open','high','low','close']
  and a DatetimeIndex (UTC preferred) when time-based features are desired.
- Set `final_fillna=True` to forward/backward fill at the end (off by default).
- Batching helps for very large frames; otherwise set `batch_size=None`.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from statsmodels.regression.linear_model import OLS

# Optional dependencies ---------------------------------------------------------
try:  # Wavelets
    import pywt  # type: ignore
    _HAS_PYWT = True
except Exception:  # pragma: no cover
    _HAS_PYWT = False

try:  # Hilbert transform
    from scipy.signal import hilbert  # type: ignore
    _HAS_HILBERT = True
except Exception:  # pragma: no cover
    _HAS_HILBERT = False

# Logging ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Return only existing columns from `cols` present in df, logging missing ones."""
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.debug(f"Missing columns skipped: {missing}")
    return existing


def batch_apply(func, df: pd.DataFrame, batch_size: int = 10000, **kwargs) -> pd.DataFrame:
    """Apply function in batches for very long dataframes (row-wise chunks)."""
    if batch_size <= 0 or len(df) <= batch_size:
        return func(df.copy(), **kwargs)
    out = []
    for i in range(0, len(df), batch_size):
        out.append(func(df.iloc[i:i + batch_size].copy(), **kwargs))
    return pd.concat(out)


def apply_with_batch(func, df: pd.DataFrame, batch_size: Optional[int] = None, **kwargs) -> pd.DataFrame:
    return batch_apply(func, df, batch_size, **kwargs) if batch_size else func(df.copy(), **kwargs)


def _infer_steps_per_day(idx: pd.DatetimeIndex) -> Optional[int]:
    """Best-effort guess of samples per day from index frequency."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return None
    step = pd.Series(idx).diff().median()
    if pd.isna(step):
        return None
    try:
        seconds = step.total_seconds()
    except Exception:
        return None
    if seconds <= 0:
        return None
    return int(round(24 * 3600 / seconds))


# --------------------------------------------------------------------------------------
# 1) Lag features
# --------------------------------------------------------------------------------------

def add_lags(df: pd.DataFrame, columns: List[str] | None = None, lags: List[int] | None = None,
             add_decay: bool = True, decay_span: int = 10) -> pd.DataFrame:
    columns = columns or _ensure_cols(df, ['close'])
    lags = lags or [1, 2, 3, 5, 10, 20, 50]
    for col in columns:
        for L in lags:
            df[f'{col}_lag_{L}'] = df[col].shift(L)
        if add_decay:
            df[f'{col}_lag_decay_{decay_span}'] = df[col].ewm(span=decay_span, adjust=False).mean().shift(1)
    return df


# --------------------------------------------------------------------------------------
# 2) Rolling statistics features
# --------------------------------------------------------------------------------------

def add_rolling_stats(
    df: pd.DataFrame,
    column: str = 'close',
    windows: List[int] | None = None,
    add_zscore: bool = True,
    add_skew_kurt: bool = True,
) -> pd.DataFrame:
    windows = windows or [5, 10, 20, 50, 100]
    s = df[column]
    for w in windows:
        r = s.rolling(window=w, min_periods=1)
        df[f'{column}_sma_{w}'] = r.mean()
        df[f'{column}_ema_{w}'] = s.ewm(span=w, adjust=False, min_periods=1).mean()
        df[f'{column}_std_{w}'] = r.std()
        df[f'{column}_min_{w}'] = r.min()
        df[f'{column}_max_{w}'] = r.max()
        if add_zscore:
            mu = df[f'{column}_sma_{w}']
            sd = df[f'{column}_std_{w}'].replace(0, np.nan)
            df[f'{column}_z_{w}'] = (s - mu) / sd
        if add_skew_kurt:
            df[f'{column}_skew_{w}'] = r.skew()
            df[f'{column}_kurt_{w}'] = r.kurt()
    return df


# --------------------------------------------------------------------------------------
# 3) Momentum & oscillators
# --------------------------------------------------------------------------------------

def add_rsi(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.DataFrame:
    delta = df[column].diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f'{column}_rsi'] = 100 - (100 / (1 + rs))
    return df


def add_stochastic(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.DataFrame:
    ll = df['low'].rolling(period, min_periods=1).min()
    hh = df['high'].rolling(period, min_periods=1).max()
    k = 100 * (df[column] - ll) / (hh - ll).replace(0, np.nan)
    df[f'{column}_stoch_k'] = k
    df[f'{column}_stoch_d'] = k.rolling(3, min_periods=1).mean()
    return df


def add_williams_r(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.DataFrame:
    hh = df['high'].rolling(period, min_periods=1).max()
    ll = df['low'].rolling(period, min_periods=1).min()
    df[f'{column}_williams_r'] = -100 * (hh - df[column]) / (hh - ll).replace(0, np.nan)
    return df


def add_roc(df: pd.DataFrame, column: str = 'close', period: int = 12) -> pd.DataFrame:
    df[f'{column}_roc_{period}'] = df[column].pct_change(periods=period)
    return df


def add_macd(df: pd.DataFrame, column: str = 'close', short: int = 12, long: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_s = df[column].ewm(span=short, adjust=False, min_periods=1).mean()
    ema_l = df[column].ewm(span=long, adjust=False, min_periods=1).mean()
    macd = ema_s - ema_l
    sig = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
    df[f'{column}_macd'] = macd
    df[f'{column}_macd_signal'] = sig
    df[f'{column}_macd_histogram'] = macd - sig
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    # Wilder's DMI/ADX
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)

    atr = pd.Series(tr, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr

    dx = (100 * (pdi - mdi).abs() / (pdi + mdi)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    df['pdi'] = pdi
    df['mdi'] = mdi
    df['adx'] = adx
    return df


# --------------------------------------------------------------------------------------
# 4) Volatility bands & channels
# --------------------------------------------------------------------------------------

def add_bollinger(df: pd.DataFrame, column: str = 'close', window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    m = df[column].rolling(window, min_periods=1).mean()
    s = df[column].rolling(window, min_periods=1).std()
    df[f'{column}_bb_mid_{window}'] = m
    df[f'{column}_bb_upper_{window}'] = m + num_std * s
    df[f'{column}_bb_lower_{window}'] = m - num_std * s
    df[f'{column}_bb_width_{window}'] = (df[f'{column}_bb_upper_{window}'] - df[f'{column}_bb_lower_{window}']) / m.replace(0, np.nan)
    df[f'{column}_bb_percent_b_{window}'] = (df[column] - df[f'{column}_bb_lower_{window}']) / (2 * num_std * s).replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    hl = (df['high'] - df['low']).abs()
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = np.nanmax(np.vstack([hl, hc, lc]), axis=0)
    df['atr'] = pd.Series(tr, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    return df


def add_keltner(df: pd.DataFrame, ema_len: int = 20, atr_mult: float = 2.0) -> pd.DataFrame:
    ema = df['close'].ewm(span=ema_len, adjust=False, min_periods=1).mean()
    if 'atr' not in df.columns:
        df = add_atr(df)
    df[f'kc_mid_{ema_len}'] = ema
    df[f'kc_upper_{ema_len}'] = ema + atr_mult * df['atr']
    df[f'kc_lower_{ema_len}'] = ema - atr_mult * df['atr']
    return df


def add_donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df[f'donchian_high_{window}'] = df['high'].rolling(window, min_periods=1).max()
    df[f'donchian_low_{window}'] = df['low'].rolling(window, min_periods=1).min()
    return df


# --------------------------------------------------------------------------------------
# 5) Derived transforms (returns, DPO, Fourier, Hilbert)
# --------------------------------------------------------------------------------------

def add_basic_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['ret_abs'] = df['ret'].abs()
    df['range_hl'] = df['high'] - df['low']
    df['range_oc'] = (df['close'] - df['open']).abs()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['dpo_20'] = df['close'] - df['close'].rolling(20, min_periods=1).mean().shift(10)
    return df


def add_fourier_terms(df: pd.DataFrame, periods: Optional[List[int]] = None, K: int = 3) -> pd.DataFrame:
    """Add Fourier seasonal terms. If periods is None, infer daily/weekly from index."""
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else None
    steps_per_day = _infer_steps_per_day(idx) if idx is not None else None
    if periods is None:
        periods = []
        if steps_per_day:
            periods.append(steps_per_day)  # daily cycle
            periods.append(steps_per_day * 7)  # weekly cycle
    n = len(df)
    t = np.arange(n)
    for P in (periods or []):
        if P is None or P == 0:
            continue
        for k in range(1, K + 1):
            df[f'f_sin_{P}_{k}'] = np.sin(2 * np.pi * k * t / P)
            df[f'f_cos_{P}_{k}'] = np.cos(2 * np.pi * k * t / P)
    return df


def add_hilbert_features(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
    if not _HAS_HILBERT:
        logger.debug('Hilbert not available; skipping')
        return df
    try:
        x = df[column].astype(float).to_numpy()
        analytic = hilbert(x)
        amplitude_envelope = np.abs(analytic)
        instantaneous_phase = np.unwrap(np.angle(analytic))
        instantaneous_freq = np.diff(instantaneous_phase, prepend=instantaneous_phase[0])
        df[f'{column}_hilbert_amp'] = amplitude_envelope
        df[f'{column}_hilbert_freq'] = instantaneous_freq
    except Exception as e:
        logger.warning(f'Hilbert features failed: {e}')
    return df


def add_forecast_residuals(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.DataFrame:
    """Rolling OLS on last `window` points, 1-step ahead forecast residual."""
    residuals = [np.nan] * len(df)
    if len(df) <= window:
        df['forecast_residual'] = np.nan
        return df
    X_base = np.arange(window).reshape(-1, 1)
    for i in range(window, len(df)):
        y = df[column].iloc[i - window:i].values
        X = np.c_[np.ones(window), X_base]
        model = OLS(y, X).fit()
        forecast = model.predict([1, window])[0]
        residuals[i] = df[column].iloc[i] - forecast
    df['forecast_residual'] = residuals
    return df


def add_convolution_patterns(df: pd.DataFrame, column: str = 'close', kernel: np.ndarray | None = None) -> pd.DataFrame:
    kernel = kernel if kernel is not None else np.array([-1, 0, 1])
    conv = np.convolve(df[column].values, kernel, mode='same')
    df[f'{column}_conv_edge'] = conv
    return df


# --------------------------------------------------------------------------------------
# 6) Session & external features
# --------------------------------------------------------------------------------------

def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        logger.warning('Index is not DatetimeIndex; session features skipped')
        return df
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    hours = idx.hour
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    df['dow_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
    # Approx session windows in UTC
    df['session_tokyo'] = ((hours >= 0) & (hours < 9)).astype(int)
    df['session_london'] = ((hours >= 7) & (hours < 16)).astype(int)
    df['session_newyork'] = ((hours >= 12) & (hours < 21)).astype(int)
    return df


def merge_external_series(df: pd.DataFrame, externals: Optional[Dict[str, pd.Series]] = None, how: str = 'left') -> pd.DataFrame:
    """Merge external time series (e.g., VIX, rates) aligned on index."""
    if not externals:
        return df
    out = df.copy()
    for name, ser in externals.items():
        s = ser.copy()
        s.name = name
        out = out.join(s, how=how)
    return out


# --------------------------------------------------------------------------------------
# 7) Wavelet denoising
# --------------------------------------------------------------------------------------

def _wavelet_denoise_series(x: np.ndarray, wavelet: str = 'db2', level: int = 1, mode: str = 'soft') -> np.ndarray:
    if not _HAS_PYWT:
        # Fallback: simple median filter
        from scipy.signal import medfilt  # type: ignore
        k = 2 * level + 1
        return medfilt(x, kernel_size=k)
    coeffs = pywt.wavedec(x, wavelet, mode='per')
    # Universal threshold
    detail_coeffs = coeffs[1:]
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745 if len(detail_coeffs[-1]) else 0.0
    uthr = sigma * np.sqrt(2 * np.log(len(x))) if sigma > 0 else 0.0
    coeffs_thresh = [coeffs[0]]
    for c in detail_coeffs:
        coeffs_thresh.append(pywt.threshold(c, value=uthr, mode=mode))
    rec = pywt.waverec(coeffs_thresh, wavelet, mode='per')
    rec = rec[: len(x)]
    return rec


def add_wavelet_denoise(df: pd.DataFrame, column: str = 'close', wavelet: str = 'db2', level: int = 1) -> pd.DataFrame:
    try:
        den = _wavelet_denoise_series(df[column].values.astype(float), wavelet=wavelet, level=level)
        df[f'{column}_denoised_wv'] = den
    except Exception as e:
        logger.warning(f'Wavelet denoising failed: {e}')
        df[f'{column}_denoised_wv'] = df[column]
    return df


# --------------------------------------------------------------------------------------
# 8) Fractional differencing (Lopez de Prado style, truncated weights)
# --------------------------------------------------------------------------------------

def _fracdiff_weights(d: float, size: int, thresh: float = 1e-4) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < size:
        w_k = -w[-1] * (d - (k - 1)) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w)


def add_fractional_diff(df: pd.DataFrame, column: str = 'close', d: float = 0.4, thresh: float = 1e-4) -> pd.DataFrame:
    w = _fracdiff_weights(d, size=len(df), thresh=thresh)
    x = df[column].values.astype(float)
    fd = np.convolve(x, w[::-1], mode='valid')
    pad = np.full(len(df) - len(fd), np.nan)
    df[f'{column}_fracdiff_{d:.2f}'] = np.concatenate([pad, fd])
    return df


def add_fractional_diff_auto(df: pd.DataFrame, column: str = 'close', d_grid: Tuple[float, float, float] = (0.2, 0.8, 0.1), pval: float = 0.05) -> pd.DataFrame:
    """Search d in grid until ADF test passes (p < pval)."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception as e:
        logger.warning(f'ADF unavailable ({e}); using default d=0.4')
        return add_fractional_diff(df, column=column, d=0.4)
    d_min, d_max, d_step = d_grid
    best = None
    for d in np.arange(d_min, d_max + 1e-9, d_step):
        tmp = add_fractional_diff(df.copy(), column=column, d=float(d))
        series = tmp[f'{column}_fracdiff_{d:.2f}'].dropna()
        if len(series) < 20:
            continue
        try:
            pv = adfuller(series)[1]
        except Exception:
            pv = 1.0
        if pv < pval:
            best = d
            break
    if best is None:
        best = 0.4
    return add_fractional_diff(df, column=column, d=best)


# --------------------------------------------------------------------------------------
# 9) Event-driven features
# --------------------------------------------------------------------------------------

def add_event_features(
    df: pd.DataFrame,
    events: Optional[Union[List[pd.Timestamp], pd.DataFrame]] = None,
    event_time_col: str = 'time',
    event_type_col: str = 'type',
    window: str = '2H',
    halflife: str = '2H',
) -> pd.DataFrame:
    """Add binary and decaying proximity-to-event features.
    - events: list of timestamps or DataFrame with time (+ optional type)
    - window: symmetric window around event time treated as active
    - halflife: exponential decay from the event time
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning('Index is not DatetimeIndex; event features skipped')
        return df
    if events is None or (isinstance(events, (list, tuple)) and len(events) == 0):
        df['event_active'] = 0
        df['event_decay'] = 0.0
        return df

    # Normalize events to DataFrame
    if isinstance(events, (list, tuple)):
        events = pd.DataFrame({event_time_col: pd.to_datetime(list(events))})
    else:
        events = events.copy()
        events[event_time_col] = pd.to_datetime(events[event_time_col])

    # Align timezones (assume df is UTC if tz-naive)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    ev_t = events[event_time_col]
    if ev_t.dt.tz is None:
        ev_t = ev_t.dt.tz_localize('UTC')

    # Active window
    win = pd.to_timedelta(window)
    half = pd.to_timedelta(halflife)

    active = np.zeros(len(df), dtype=int)
    decay = np.zeros(len(df), dtype=float)

    for t in ev_t:
        m = (idx >= (t - win)) & (idx <= (t + win))
        active = np.maximum(active, m.astype(int))
        dt = (idx - t).total_seconds().to_numpy()
        lam = np.log(2) / half.total_seconds() if half.total_seconds() > 0 else 0.0
        dec = np.exp(-lam * np.abs(dt))
        decay = np.maximum(decay, dec)

    df['event_active'] = active
    df['event_decay'] = decay

    # Optional: one-hot for event types
    if event_type_col in events.columns:
        for typ, tseries in events.groupby(event_type_col)[event_time_col]:
            col = f'event_{str(typ).lower()}_active'
            m = np.zeros(len(df), dtype=int)
            for t in pd.to_datetime(tseries):
                m = np.maximum(m, ((idx >= (t - win)) & (idx <= (t + win))).astype(int))
            df[col] = m

    # Pre/post realized return windows (impact features)
    for h in [1, 3, 6]:
        df[f'ret_post_event_{h}h'] = df['close'].pct_change(h).shift(-h) * df['event_active']
        df[f'ret_pre_event_{h}h'] = df['close'].pct_change(h) * df['event_active']

    return df


# --------------------------------------------------------------------------------------
# 10) Lorentzian classification (KDE on normalized multi-feature space)
# --------------------------------------------------------------------------------------

def add_lorentzian_classification(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    bandwidth: float = 0.1,
) -> pd.DataFrame:
    defaults = [
        'close_rsi', 'close_macd_histogram', 'ret', 'log_return', 'ema_12', 'ema_26',
        'close_bb_percent_b_20', 'adx', 'pdi', 'mdi'
    ]
    features = features or _ensure_cols(df, defaults)
    features = _ensure_cols(df, features)
    if not features:
        logger.warning('Lorentzian: no features available; skipping')
        df['lorentz_buy_prob'] = 0.0
        df['lorentz_sell_prob'] = 0.0
        df['lorentz_hold_prob'] = 1.0
        return df

    norm = (df[features] - df[features].mean()) / df[features].std(ddof=0)
    norm = norm.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    try:
        kde = gaussian_kde(norm.T, bw_method=bandwidth)
        dens = kde(norm.T)
    except Exception as e:
        logger.warning(f'KDE failed ({e}); using uniform density')
        dens = np.ones(len(df))
    dist = np.linalg.norm(norm.diff().fillna(0.0).values, axis=1)
    med = np.nanmedian(dist)
    up = df['close'].diff().fillna(0.0) > 0
    down = ~up

    scale = np.max(dens) if np.max(dens) > 0 else 1.0
    df['lorentz_buy_prob'] = np.where((dist > med) & up, dens / scale, 0.0)
    df['lorentz_sell_prob'] = np.where((dist > med) & down, dens / scale, 0.0)
    df['lorentz_hold_prob'] = 1.0 - (df['lorentz_buy_prob'] + df['lorentz_sell_prob'])
    return df


# --------------------------------------------------------------------------------------
# Orchestration pipeline
# --------------------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    *,
    batch_size: Optional[int] = None,
    config: Optional[Dict] = None,
    events: Optional[Union[List[pd.Timestamp], pd.DataFrame]] = None,
    externals: Optional[Dict[str, pd.Series]] = None,
    final_fillna: bool = False,
) -> pd.DataFrame:
    if len(df) < 50:
        raise ValueError('Data too short for robust feature engineering (min 50 rows).')
    required = _ensure_cols(df, ['open', 'high', 'low', 'close'])
    if len(required) < 4:
        raise ValueError('Missing one of required columns: open, high, low, close')

    logger.info(f'Engineering on shape {df.shape}')

    # Defaults
    default_cfg = dict(
        lags=dict(columns=['close'], lags=[1, 2, 3, 5, 10, 20, 50], add_decay=True, decay_span=10),
        rolling_stats=dict(column='close', windows=[5, 10, 20, 50, 100], add_zscore=True, add_skew_kurt=True),
        oscillators=dict(rsi=14, stoch=14, roc=12, macd=(12, 26, 9), adx=14),
        vol_bands=dict(bb_window=20, bb_std=2.0, keltner_ema=20, keltner_mult=2.0, donchian=20),
        transforms=True,
        fourier=dict(periods=None, K=3),
        hilbert=True,
        forecast_residuals=20,
        conv_kernel=[-1, 0, 1],
        sessions=True,
        externals=True,
        wavelet=dict(wavelet='db2', level=1),
        fracdiff=dict(mode='auto', d=0.4, grid=(0.2, 0.8, 0.1), pval=0.05),
        events=dict(window='2H', halflife='2H'),
        lorentzian=dict(bandwidth=0.1),
        select_features=False,
    )
    cfg = {**default_cfg, **(config or {})}

    # 0) External series merge (if any)
    if cfg.get('externals') and externals:
        df = merge_external_series(df, externals=externals)

    # 1) Lags
    df = apply_with_batch(add_lags, df, batch_size, **cfg['lags'])

    # 2) Rolling stats
    df = apply_with_batch(add_rolling_stats, df, batch_size, **cfg['rolling_stats'])

    # 3) Momentum & oscillators
    df = apply_with_batch(add_rsi, df, batch_size, column='close', period=cfg['oscillators']['rsi'])
    df = apply_with_batch(add_stochastic, df, batch_size, period=cfg['oscillators']['stoch'])
    df = apply_with_batch(add_williams_r, df, batch_size, period=cfg['oscillators']['stoch'])
    df = apply_with_batch(add_roc, df, batch_size, column='close', period=cfg['oscillators']['roc'])
    macd_s, macd_l, macd_sig = cfg['oscillators']['macd']
    df = apply_with_batch(add_macd, df, batch_size, column='close', short=macd_s, long=macd_l, signal=macd_sig)
    df = apply_with_batch(add_adx, df, batch_size, period=cfg['oscillators']['adx'])

    # 4) Volatility bands / channels
    df = apply_with_batch(add_bollinger, df, batch_size, column='close', window=cfg['vol_bands']['bb_window'], num_std=cfg['vol_bands']['bb_std'])
    df = apply_with_batch(add_atr, df, batch_size, period=14)
    df = apply_with_batch(add_keltner, df, batch_size, ema_len=cfg['vol_bands']['keltner_ema'], atr_mult=cfg['vol_bands']['keltner_mult'])
    df = apply_with_batch(add_donchian, df, batch_size, window=cfg['vol_bands']['donchian'])

    # 5) Derived transforms
    if cfg['transforms']:
        df = apply_with_batch(add_basic_transforms, df, batch_size)
    if cfg.get('fourier'):
        df = apply_with_batch(add_fourier_terms, df, batch_size, **cfg['fourier'])
    if cfg.get('hilbert'):
        df = apply_with_batch(add_hilbert_features, df, batch_size, column='close')

    # 6) Forecast residuals & simple CNN-like edge
    if cfg['forecast_residuals']:
        df = apply_with_batch(add_forecast_residuals, df, batch_size, column='close', window=cfg['forecast_residuals'])
    if cfg['conv_kernel'] is not None:
        df = apply_with_batch(add_convolution_patterns, df, batch_size, column='close', kernel=np.array(cfg['conv_kernel']))

    # 7) Session & external
    if cfg['sessions']:
        df = apply_with_batch(add_session_features, df, batch_size)

    # 8) Wavelet denoising
    if cfg['wavelet']:
        df = apply_with_batch(add_wavelet_denoise, df, batch_size, column='close', **cfg['wavelet'])

    # 9) Fractional differencing (auto or fixed)
    frac_cfg = cfg.get('fracdiff')
    if frac_cfg:
        if isinstance(frac_cfg, dict) and frac_cfg.get('mode', 'fixed') == 'auto':
            df = apply_with_batch(add_fractional_diff_auto, df, batch_size, column='close', d_grid=frac_cfg.get('grid', (0.2, 0.8, 0.1)), pval=frac_cfg.get('pval', 0.05))
        else:
            d_val = frac_cfg['d'] if isinstance(frac_cfg, dict) else 0.4
            df = apply_with_batch(add_fractional_diff, df, batch_size, column='close', d=d_val)

    # 10) Event-driven
    if cfg.get('events') is not None:
        df = apply_with_batch(add_event_features, df, batch_size, events=events, **cfg['events'])

    # 11) Lorentzian classification
    df = apply_with_batch(add_lorentzian_classification, df, batch_size, **cfg['lorentzian'])

    # Final touches
    if final_fillna:
        df = df.fillna(method='ffill').fillna(method='bfill')

    logger.info(f'Feature engineering completed. Final shape: {df.shape}')
    return df


# --------------------------------------------------------------------------------------
# Example / self-test (remove or guard in production)
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    rng = pd.date_range('2025-08-01 00:00', periods=500, freq='H', tz='UTC')
    sample_df = pd.DataFrame({
        'open': np.random.uniform(1.08, 1.12, len(rng)),
        'high': np.random.uniform(1.08, 1.12, len(rng)),
        'low': np.random.uniform(1.07, 1.11, len(rng)),
        'close': np.random.uniform(1.075, 1.115, len(rng)),
    }, index=rng)

    # Example macro events (e.g., CPI, FOMC)
    ev_times = [pd.Timestamp('2025-08-03 12:30Z'), pd.Timestamp('2025-08-10 18:00Z')]

    # Example external series (e.g., VIX proxy)
    externals = {
        'vix_proxy': pd.Series(np.random.uniform(12, 24, len(rng)), index=rng),
        'us10y_proxy': pd.Series(np.random.uniform(3.5, 5.0, len(rng)), index=rng),
    }

    feats = engineer_features(sample_df, batch_size=None, events=ev_times, externals=externals, final_fillna=False)
    pd.set_option('display.max_columns', None)
    print(feats.head())
