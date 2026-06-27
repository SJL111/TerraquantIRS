"""
core/price_data.py — market data with dual backend:

  Primary  : Futu OpenD (富途本地行情网关, localhost:11111)
  Fallback : yfinance  (used automatically when OpenD is unavailable)

This means the app works both:
  • Locally with Futu OpenD running  → real-time / pre-adjusted data
  • On Streamlit Cloud / any server   → yfinance public data
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# ── Detect Futu availability (import-time check) ──────────────────────────────
try:
    from futu import AuType, KLType, OpenQuoteContext, RET_OK as _RET_OK
    _FUTU_IMPORTED = True
except Exception:
    _FUTU_IMPORTED = False

FUTU_HOST = "127.0.0.1"
FUTU_PORT = 11111


def _futu_open():
    """Open a Futu quote context; return None if OpenD is unreachable."""
    if not _FUTU_IMPORTED:
        return None
    try:
        ctx = OpenQuoteContext(host=FUTU_HOST, port=FUTU_PORT)
        return ctx
    except Exception:
        return None


def to_futu_code(ticker: str) -> str | None:
    t = ticker.strip()
    if t.endswith(".HK"):
        return f"HK.{t[:-3].zfill(5)}"
    if t.endswith(".KS") or t.endswith(".TW"):
        return None
    if "." in t:
        return None
    return f"US.{t}"


# ── yfinance helpers ──────────────────────────────────────────────────────────

def _yf_period(period: str) -> str:
    """Translate our period strings to yfinance period strings."""
    mapping = {"1y": "1y", "3y": "3y", "5y": "5y", "10y": "10y", "max": "max"}
    return mapping.get(period, "3y")


# ── Price history ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    """
    Daily OHLCV. Tries Futu OpenD first; falls back to yfinance automatically.
    """
    # ── Try Futu ──────────────────────────────────────────────────────────────
    ctx = _futu_open()
    if ctx is not None:
        try:
            code   = to_futu_code(ticker)
            years  = int(period.replace("y", "")) if period.endswith("y") else 3
            start  = (datetime.today() - timedelta(days=years * 366)).strftime("%Y-%m-%d")
            end    = datetime.today().strftime("%Y-%m-%d")
            chunks, page_key = [], None
            while True:
                kwargs = dict(
                    code=code, start=start, end=end,
                    ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=1000,
                )
                if page_key is not None:
                    kwargs["page_req_key"] = page_key
                ret, data, page_key = ctx.request_history_kline(**kwargs)
                if ret != _RET_OK or data is None or data.empty:
                    break
                chunks.append(data)
                if page_key is None:
                    break
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                df["time_key"] = pd.to_datetime(df["time_key"])
                df = df.set_index("time_key").sort_index()
                df.index.name = "Date"
                df = df.rename(columns={
                    "open": "Open", "high": "High",
                    "low":  "Low",  "close": "Close", "volume": "Volume",
                })
                return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            pass
        finally:
            ctx.close()

    # ── Fallback: yfinance ────────────────────────────────────────────────────
    try:
        import yfinance as yf
        yf_period = _yf_period(period)
        df = yf.Ticker(ticker).history(period=yf_period, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna()
    except Exception:
        return pd.DataFrame()


# ── Snapshot / info ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_snapshot(ticker: str) -> dict:
    """Futu real-time snapshot; returns {} if unavailable."""
    ctx = _futu_open()
    if ctx is None:
        return {}
    code = to_futu_code(ticker)
    if not code:
        ctx.close()
        return {}
    try:
        ret, data = ctx.get_market_snapshot([code])
        if ret != _RET_OK or data is None or data.empty:
            return {}
        return data.iloc[0].to_dict()
    except Exception:
        return {}
    finally:
        ctx.close()


@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker: str) -> dict:
    """
    Company info dict. Tries Futu snapshot, falls back to yfinance.
    Maintains yfinance key names for backward compatibility.
    """
    snap = get_snapshot(ticker)
    if snap:
        return {
            "longName":          snap.get("name", ticker),
            "symbol":            ticker,
            "marketCap":         snap.get("total_market_val"),
            "trailingPE":        snap.get("pe_ttm_ratio"),
            "priceToBook":       snap.get("pb_ratio"),
            "trailingEps":       snap.get("earning_per_share"),
            "dividendYield":     snap.get("dividend_ratio_ttm"),
            "fiftyTwoWeekHigh":  snap.get("highest52weeks_price"),
            "fiftyTwoWeekLow":   snap.get("lowest52weeks_price"),
            "currentPrice":      snap.get("last_price"),
            "beta":              None,
        }

    # Fallback: yfinance
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


# ── Quarterly returns ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_quarterly_returns(ticker: str) -> pd.Series:
    px = get_price_history(ticker, period="5y")
    if px.empty:
        return pd.Series(dtype=float)
    q = px["Close"].resample("QE").last()
    return q.pct_change().dropna()


# ── Key metrics display ───────────────────────────────────────────────────────

def _fmt(val, fmt: str = "{:.2f}", suffix: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        return fmt.format(float(val)) + suffix
    except Exception:
        return "N/A"


def _fmt_bn(val) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1e12:
            return f"{v/1e12:.2f} T"
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f} B"
        if abs(v) >= 1e6:
            return f"{v/1e6:.2f} M"
        return str(v)
    except Exception:
        return "N/A"


def key_metrics(ticker: str) -> dict:
    """Formatted metrics dict for st.metric cards."""
    snap = get_snapshot(ticker)
    if snap:
        return {
            "最新价":       _fmt(snap.get("last_price"), "{:.2f}", " USD"),
            "市值":         _fmt_bn(snap.get("total_market_val")),
            "PE (TTM)":     _fmt(snap.get("pe_ttm_ratio"), "{:.1f}"),
            "PB":           _fmt(snap.get("pb_ratio"), "{:.2f}"),
            "EPS":          _fmt(snap.get("earning_per_share"), "{:.2f}", " USD"),
            "股息率 (TTM)": _fmt(snap.get("dividend_ratio_ttm"), "{:.2f}", "%"),
            "52W 高":       _fmt(snap.get("highest52weeks_price"), "{:.2f}", " USD"),
            "52W 低":       _fmt(snap.get("lowest52weeks_price"), "{:.2f}", " USD"),
            "换手率":       _fmt(snap.get("turnover_rate"), "{:.2f}", "%"),
            "量比":         _fmt(snap.get("volume_ratio"), "{:.2f}"),
            "涨跌幅":       _fmt(snap.get("change_rate"), "{:.2f}", "%"),
        }

    # Fallback: yfinance info
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        fi   = yf.Ticker(ticker).fast_info
        price = getattr(fi, "last_price", None) or info.get("currentPrice")
        return {
            "最新价":       _fmt(price, "{:.2f}", " USD"),
            "市值":         _fmt_bn(info.get("marketCap")),
            "PE (TTM)":     _fmt(info.get("trailingPE"), "{:.1f}"),
            "PB":           _fmt(info.get("priceToBook"), "{:.2f}"),
            "EPS":          _fmt(info.get("trailingEps"), "{:.2f}", " USD"),
            "股息率 (TTM)": _fmt(info.get("dividendYield"), "{:.2%}"),
            "52W 高":       _fmt(info.get("fiftyTwoWeekHigh"), "{:.2f}", " USD"),
            "52W 低":       _fmt(info.get("fiftyTwoWeekLow"), "{:.2f}", " USD"),
            "Beta":         _fmt(info.get("beta"), "{:.2f}"),
            "行业":         info.get("industry", "N/A"),
            "来源":         "yfinance (Futu 未连接)",
        }
    except Exception:
        return {"状态": "价格数据不可用"}
