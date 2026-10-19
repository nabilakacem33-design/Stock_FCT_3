"""Data loading: prices, market context, company info, news."""
import pandas as pd
import streamlit as st
import yfinance as yf


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    return _clean(df) if df is not None and not df.empty else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_market(years: int) -> pd.DataFrame:
    """S&P 500 (SPY) and VIX as market-wide context features."""
    out = pd.DataFrame()
    try:
        spy = _clean(yf.download("SPY", period=f"{years}y", auto_adjust=True, progress=False))
        vix = _clean(yf.download("^VIX", period=f"{years}y", progress=False))
        out["spy"] = spy["Close"]
        out["vix"] = vix["Close"]
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_news(ticker: str) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    items = []
    for it in raw:
        c = it.get("content", it)
        title = c.get("title") or ""
        if not title:
            continue
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or ""
        if c.get("pubDate"):
            date = pd.to_datetime(c["pubDate"], errors="coerce")
        else:
            date = pd.to_datetime(c.get("providerPublishTime"), unit="s", errors="coerce")
        items.append({
            "title": title,
            "summary": c.get("summary") or "",
            "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher") or "",
            "date": date,
            "url": url,
        })
    return items
