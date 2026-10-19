"""Technical + market-context feature engineering."""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    lc = np.log(c)
    r = lc.diff()
    d = pd.DataFrame(index=df.index)

    for k in (1, 2, 3, 5, 10, 20, 60):
        d[f"ret_{k}"] = lc.diff(k)
    d["vol_10"], d["vol_20"] = r.rolling(10).std(), r.rolling(20).std()
    d["vol_ratio"] = d["vol_10"] / d["vol_20"]
    for w in (10, 20, 50, 200):
        d[f"sma_{w}"] = c / c.rolling(w).mean() - 1

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    macd = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    d["macd"] = macd
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    m, s = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_pos"] = (c - m) / (2 * s)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean() / c
    d["range"] = (h - l) / c
    d["vol_chg"] = np.log(v.replace(0, np.nan) / v.rolling(20).mean())
    d["dist_52w_high"] = c / c.rolling(252, min_periods=60).max() - 1
    d["dow"] = df.index.dayofweek

    if mkt is not None and not mkt.empty:
        mk = mkt.reindex(df.index).ffill()
        if "spy" in mk:
            d["spy_ret_5"] = np.log(mk["spy"]).diff(5)
            d["spy_ret_20"] = np.log(mk["spy"]).diff(20)
        if "vix" in mk:
            d["vix"] = mk["vix"]
            d["vix_chg_5"] = mk["vix"].pct_change(5)

    return d.replace([np.inf, -np.inf], np.nan)
