"""Direct multi-horizon forecasting (ensemble + quantile bands) and walk-forward backtest."""
import math

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]
SENTIMENT_STRENGTH = 0.15  # max tilt = 0.15 x one-sigma move


def _hgb(q=None):
    kw = dict(max_iter=120, learning_rate=0.05, max_depth=3, min_samples_leaf=30,
              l2_regularization=1.0, random_state=0)
    if q is None:
        return HistGradientBoostingRegressor(**kw)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw)


def _fit_mean(X, y):
    """Ensemble of gradient boosting + ridge; returns a predict function."""
    hgb = _hgb().fit(X, y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=50.0)).fit(X.fillna(0), y)
    return lambda Z: 0.5 * hgb.predict(Z) + 0.5 * ridge.predict(Z.fillna(0))


def _target(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h) / close)


@st.cache_data(ttl=3600, show_spinner=False)
def forecast(close: pd.Series, feats: pd.DataFrame, horizon: int,
             sentiment: float = 0.0, use_sentiment: bool = True) -> pd.DataFrame:
    last_x = feats.iloc[[-1]]
    daily_vol = float(np.log(close).diff().tail(60).std())
    hs = [h for h in GRID if h < horizon] + [horizon]
    rows = []
    for h in hs:
        data = feats.join(_target(close, h).rename("y")).dropna()
        X, y = data.drop(columns="y"), data["y"]
        scale = daily_vol * math.sqrt(h)
        mu = float(np.clip(_fit_mean(X, y)(last_x)[0], -4 * scale, 4 * scale))
        lo = _hgb(0.1).fit(X, y).predict(last_x)[0]
        hi = _hgb(0.9).fit(X, y).predict(last_x)[0]
        lo, hi = min(lo, mu - 0.5 * scale), max(hi, mu + 0.5 * scale)  # keep band non-degenerate
        if use_sentiment:
            shift = SENTIMENT_STRENGTH * sentiment * scale
            mu, lo, hi = mu + shift, lo + shift, hi + shift
        rows.append((h, mu, lo, hi))
    f = pd.DataFrame(rows, columns=["h", "mu", "lo", "hi"]).set_index("h")
    days = np.arange(1, horizon + 1)
    out = {c: np.interp(days, f.index, f[c]) for c in f.columns}
    last = float(close.iloc[-1])
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({
        "date": dates,
        "price": last * np.exp(out["mu"]),
        "low": last * np.exp(out["lo"]),
        "high": last * np.exp(out["hi"]),
        "ret": out["mu"],
        "sigma": (out["hi"] - out["lo"]) / 2.563,
    })


def prob_up(ret: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf(ret / (max(sigma, 1e-9) * math.sqrt(2))))


@st.cache_data(ttl=3600, show_spinner=False)
def backtest(close: pd.Series, feats: pd.DataFrame, h: int, n_test: int = 300, step: int = 10):
    """Walk-forward (expanding window, no look-ahead) out-of-sample test."""
    data = feats.join(_target(close, h).rename("y")).dropna()
    n_test = min(n_test, len(data) // 3)
    start = len(data) - n_test
    parts = []
    for i in range(start, len(data), step):
        train = data.iloc[: i - h]  # drop last h rows: their targets overlap the test period
        test = data.iloc[i: i + step]
        if len(train) < 250 or test.empty:
            continue
        pred = _fit_mean(train.drop(columns="y"), train["y"])(test.drop(columns="y"))
        parts.append(pd.DataFrame({"pred": pred, "actual": test["y"]}, index=test.index))
    if not parts:
        return None, {}
    bt = pd.concat(parts)
    strat = np.sign(bt.pred) * bt.actual
    stats = {
        "hit": float((np.sign(bt.pred) == np.sign(bt.actual)).mean()),
        "base": float(max((bt.actual > 0).mean(), (bt.actual <= 0).mean())),
        "mae": float((bt.pred - bt.actual).abs().mean()),
        "mae0": float(bt.actual.abs().mean()),
        "ic": float(bt.pred.corr(bt.actual)),
        "n": len(bt),
    }
    return bt, stats
