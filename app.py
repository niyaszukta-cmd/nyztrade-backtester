"""
NYZTrade SMC Liquidity Lens Backtester
Index Options Backtest | Dhan API | Built for NIYAS
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NYZTrade | SMC Liquidity Lens",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div[data-testid="metric-container"] {
        background: #1a1d29; border-radius: 8px;
        padding: 12px; border: 1px solid #2d3139;
    }
    .sidebar-header {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #666;
        margin-top: 1.2rem; margin-bottom: 0.3rem;
    }
    .token-box {
        background: #1a1d29; border: 1px solid #2d3139;
        border-radius: 8px; padding: 12px; margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1d29; border-radius: 6px; border: 1px solid #2d3139;
    }
    .stTabs [aria-selected="true"] { background: #0052cc; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DHAN CONFIG — identical pattern to GEX Dashboard
# Update access_token daily. client_id never changes.
# ══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass

@dataclass
class DhanConfig:
    client_id:    str = "1100480354"
    access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcyMTY3OTgxLCJhcHBfaWQiOiJjOTNkM2UwOSIsImlhdCI6MTc3MjA4MTU4MSwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.Kry8jyKMhIR-f1H5R0a2A4I9UHnWdDDE3LMmnXgOiE2U5pXWP3P0Scohw4j4IPvBPy3bPienE2vrWdU78bdJ0w"   # ← paste fresh token here daily

DHAN_BASE = "https://api.dhan.co/v2"

INDICES = {
    "NIFTY 50":   {"security_id": 13,  "lot_size": 75,  "strike_gap": 50},
    "BANKNIFTY":  {"security_id": 25,  "lot_size": 30,  "strike_gap": 100},
    "FINNIFTY":   {"security_id": 27,  "lot_size": 65,  "strike_gap": 50},
    "MIDCPNIFTY": {"security_id": 442, "lot_size": 75,  "strike_gap": 25},
    "SENSEX":     {"security_id": 1,   "lot_size": 10,  "strike_gap": 100},
    "BANKEX":     {"security_id": 12,  "lot_size": 15,  "strike_gap": 100},
}


# ══════════════════════════════════════════════════════════════════════════════
# DHAN FETCHER CLASS — mirrors GEX Dashboard's UnifiedOptionsFetcher exactly
# No token passed anywhere. self.headers used on every request.
# ══════════════════════════════════════════════════════════════════════════════

class DhanFetcher:
    def __init__(self, config: DhanConfig):
        self.config = config
        # Identical header structure to GEX Dashboard
        self.headers = {
            "access-token": config.access_token,
            "client-id":    config.client_id,
            "Content-Type": "application/json",
        }
        self.base_url = DHAN_BASE

    def _handle_error(self, r: requests.Response) -> None:
        code, body = r.status_code, r.text[:400]
        if code == 401:
            st.error("❌ **401** — Token expired/invalid. Update `access_token` in DhanConfig and restart.")
        elif code == 429:
            st.error("❌ **429** — Rate limit hit. Wait 60 s and retry.")
        elif code == 400:
            st.error(f"❌ **400 Bad Request** — `{body}`")
        else:
            st.error(f"❌ **HTTP {code}** — `{body}`")

    def fetch_index_ohlcv(self, security_id: int, from_date: str,
                           to_date: str, interval: str,
                           debug: bool = False) -> pd.DataFrame | None:
        """
        Fetch index OHLCV — same approach as GEX dashboard fetch_rolling_data.

        Intraday (1/5/15/25/60):  POST /charts/intraday
                                  exchangeSegment=IDX_I, instrument=INDEX
                                  fromDate/toDate include time  HH:MM:SS

        Daily (D):                POST /charts/historical
                                  exchangeSegment=IDX_I, instrument=INDEX
        """
        intraday_set = {"1", "5", "15", "25", "60"}

        if interval in intraday_set:
            from_dt    = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt      = datetime.strptime(to_date,   "%Y-%m-%d")
            total_days = max((to_dt - from_dt).days, 1)
            all_dfs, fetched = [], 0
            prog = st.progress(0, text="Fetching intraday data…")
            cur  = from_dt

            while cur <= to_dt:
                end = min(cur + timedelta(days=90), to_dt)
                payload = {
                    "securityId":      security_id,
                    "exchangeSegment": "IDX_I",
                    "instrument":      "INDEX",
                    "interval":        interval,
                    "fromDate":        cur.strftime("%Y-%m-%d") + " 09:15:00",
                    "toDate":          end.strftime("%Y-%m-%d") + " 15:30:00",
                }
                if debug:
                    st.write("📤 Intraday payload:", payload)
                try:
                    r = requests.post(f"{self.base_url}/charts/intraday",
                                      headers=self.headers, json=payload, timeout=30)
                    if debug:
                        st.write(f"📥 Response {r.status_code}:", r.text[:600])
                    if not r.ok:
                        self._handle_error(r)
                        prog.empty()
                        return None
                    data = r.json()
                    if isinstance(data, dict) and "data" in data:
                        data = data["data"]
                    df = _parse_ohlcv(data)
                    if df is not None:
                        all_dfs.append(df)
                except Exception as e:
                    st.error(f"Request error: {e}")
                    prog.empty()
                    return None

                fetched += max((end - cur).days, 1)
                prog.progress(min(fetched / total_days, 1.0),
                              text=f"Fetched {cur.date()} → {end.date()}")
                cur = end + timedelta(days=1)
                time.sleep(0.3)

            prog.empty()
            if not all_dfs:
                st.warning("⚠️ No data returned. Try Daily timeframe or shorter date range.")
                return None
            return (pd.concat(all_dfs)
                      .drop_duplicates("timestamp")
                      .sort_values("timestamp")
                      .reset_index(drop=True))

        else:
            payload = {
                "securityId":      security_id,
                "exchangeSegment": "IDX_I",
                "instrument":      "INDEX",
                "fromDate":        from_date,
                "toDate":          to_date,
                "expiryCode":      0,
            }
            if debug:
                st.write("📤 Daily payload:", payload)
            try:
                r = requests.post(f"{self.base_url}/charts/historical",
                                  headers=self.headers, json=payload, timeout=30)
                if debug:
                    st.write(f"📥 Response {r.status_code}:", r.text[:600])
                if not r.ok:
                    self._handle_error(r)
                    return None
                data = r.json()
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                return _parse_ohlcv(data)
            except Exception as e:
                st.error(f"Request error: {e}")
            return None

    def fetch_rolling_option(self, security_id: int, strike_offset: int,
                              opt_type: str, from_date: str, to_date: str,
                              interval: str = "25", expiry_flag: str = "WEEK",
                              debug: bool = False) -> pd.DataFrame | None:
        """
        Fetch real option OHLCV via /charts/rollingoption — chunked in 89-day windows
        to avoid DH-905 (max 90 days per call).
        opt_type: "CALL" or "PUT"
        """
        strike_str = "ATM" if strike_offset == 0 else (
            f"ATM+{strike_offset}" if strike_offset > 0 else f"ATM{strike_offset}"
        )
        key = "ce" if opt_type == "CALL" else "pe"

        from_dt    = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt      = datetime.strptime(to_date,   "%Y-%m-%d")
        total_days = max((to_dt - from_dt).days, 1)
        all_dfs, fetched = [], 0
        prog = st.progress(0, text=f"Fetching {strike_str} {opt_type} option data…")
        cur  = from_dt

        while cur <= to_dt:
            end = min(cur + timedelta(days=89), to_dt)   # max 89 days per call (DH-905)
            payload = {
                "exchangeSegment": "NSE_FNO",
                "interval":        interval,
                "securityId":      security_id,
                "instrument":      "OPTIDX",
                "expiryFlag":      expiry_flag,
                "expiryCode":      1,
                "strike":          strike_str,
                "drvOptionType":   opt_type,
                "requiredData":    ["open", "high", "low", "close", "volume", "oi"],
                "fromDate":        cur.strftime("%Y-%m-%d"),
                "toDate":          end.strftime("%Y-%m-%d"),
            }
            if debug:
                st.write(f"📤 Rolling option payload ({cur.date()} → {end.date()}):", payload)
            try:
                r = requests.post(f"{self.base_url}/charts/rollingoption",
                                  headers=self.headers, json=payload, timeout=30)
                if debug:
                    st.write(f"📥 Response {r.status_code}:", r.text[:600])
                if not r.ok:
                    self._handle_error(r)
                    prog.empty()
                    return None
                raw = r.json().get("data", {}).get(key, {})
                if raw and raw.get("timestamp"):
                    chunk = pd.DataFrame({
                        "timestamp": pd.to_datetime(raw["timestamp"], unit="s", utc=True)
                                       .tz_convert("Asia/Kolkata").tz_localize(None),
                        "open":   pd.to_numeric(raw.get("open",   []), errors="coerce"),
                        "high":   pd.to_numeric(raw.get("high",   []), errors="coerce"),
                        "low":    pd.to_numeric(raw.get("low",    []), errors="coerce"),
                        "close":  pd.to_numeric(raw.get("close",  []), errors="coerce"),
                        "volume": pd.to_numeric(raw.get("volume", []), errors="coerce"),
                        "oi":     pd.to_numeric(raw.get("oi", [np.nan]*len(raw["timestamp"])), errors="coerce"),
                    }).dropna(subset=["close"])
                    if not chunk.empty:
                        all_dfs.append(chunk)
            except Exception as e:
                st.error(f"Rolling option chunk error: {e}")
                prog.empty()
                return None

            fetched += max((end - cur).days, 1)
            prog.progress(min(fetched / total_days, 1.0),
                          text=f"Option data: {cur.date()} → {end.date()}")
            cur = end + timedelta(days=1)
            time.sleep(0.3)

        prog.empty()
        if not all_dfs:
            st.warning("⚠️ No option data returned for this strike/range.")
            return None
        return (pd.concat(all_dfs)
                  .drop_duplicates("timestamp")
                  .sort_values("timestamp")
                  .reset_index(drop=True))


# Module-level fetcher instance — uses hardcoded DhanConfig
_fetcher = DhanFetcher(DhanConfig())


def _parse_ohlcv(data: dict) -> pd.DataFrame | None:
    if not data or "open" not in data or not data.get("timestamp"):
        return None
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True)
                       .tz_convert("Asia/Kolkata").tz_localize(None),
        "open":   pd.to_numeric(data["open"],   errors="coerce"),
        "high":   pd.to_numeric(data["high"],   errors="coerce"),
        "low":    pd.to_numeric(data["low"],    errors="coerce"),
        "close":  pd.to_numeric(data["close"],  errors="coerce"),
        "volume": pd.to_numeric(data["volume"], errors="coerce"),
    }).dropna().sort_values("timestamp").reset_index(drop=True)
    return df if not df.empty else None


# ══════════════════════════════════════════════════════════════════════════════
# STRIKE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_atm_strike(spot: float, strike_gap: int) -> int:
    """Round spot to nearest strike_gap."""
    return int(round(spot / strike_gap) * strike_gap)


def get_strike_label(offset: int) -> str:
    if offset == 0:   return "ATM"
    if offset > 0:    return f"ATM+{offset}"
    return f"ATM{offset}"


def compute_strike_price(spot: float, strike_gap: int, offset: int) -> int:
    """ATM + offset * strike_gap."""
    atm = get_atm_strike(spot, strike_gap)
    return atm + offset * strike_gap


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def calc_bsp(df: pd.DataFrame, length: int) -> pd.Series:
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    ad = ((2 * df["close"] - df["low"] - df["high"]) / hl) * df["volume"]
    return (ad.rolling(length).sum() / df["volume"].rolling(length).sum()).fillna(0)


def pivot_highs(df: pd.DataFrame, n: int) -> list[dict]:
    out = []
    for i in range(n, len(df) - n):
        win = df["high"].iloc[i - n: i + n + 1]
        if df["high"].iloc[i] == win.max():
            out.append({"idx": i, "price": df["high"].iloc[i],
                        "open": df["open"].iloc[i], "close": df["close"].iloc[i],
                        "volume": df["volume"].iloc[i]})
    return out


def pivot_lows(df: pd.DataFrame, n: int) -> list[dict]:
    out = []
    for i in range(n, len(df) - n):
        win = df["low"].iloc[i - n: i + n + 1]
        if df["low"].iloc[i] == win.min():
            out.append({"idx": i, "price": df["low"].iloc[i],
                        "open": df["open"].iloc[i], "close": df["close"].iloc[i],
                        "volume": df["volume"].iloc[i]})
    return out


def order_blocks(df: pd.DataFrame, ph: list, pl: list,
                 vt: float = 1.5) -> list[dict]:
    vsma = df["volume"].rolling(14).mean()
    obs  = []
    for p in ph:
        i      = p["idx"]
        strong = p["volume"] > vsma.iloc[i] * vt if pd.notna(vsma.iloc[i]) else False
        end    = next((j for j in range(i + 1, len(df))
                       if df["close"].iloc[j] > p["price"]), len(df) - 1)
        obs.append({"type": "bearish", "top": p["price"],
                    "btm": max(p["open"], p["close"]),
                    "start": i, "end": end, "strong": strong})
    for p in pl:
        i      = p["idx"]
        strong = p["volume"] > vsma.iloc[i] * vt if pd.notna(vsma.iloc[i]) else False
        end    = next((j for j in range(i + 1, len(df))
                       if df["close"].iloc[j] < p["price"]), len(df) - 1)
        obs.append({"type": "bullish", "top": min(p["open"], p["close"]),
                    "btm": p["price"],
                    "start": i, "end": end, "strong": strong})
    return obs


def fair_value_gaps(df: pd.DataFrame) -> list[dict]:
    fvgs = []
    for i in range(2, len(df)):
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            mid = (df["low"].iloc[i] + df["high"].iloc[i - 2]) / 2
            if (df["low"].iloc[i] - df["high"].iloc[i - 2]) / mid > 0.001:
                fvgs.append({"type": "bullish", "top": df["low"].iloc[i],
                              "btm": df["high"].iloc[i - 2], "start": i - 2})
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            mid = (df["low"].iloc[i - 2] + df["high"].iloc[i]) / 2
            if (df["low"].iloc[i - 2] - df["high"].iloc[i]) / mid > 0.001:
                fvgs.append({"type": "bearish", "top": df["low"].iloc[i - 2],
                              "btm": df["high"].iloc[i], "start": i - 2})
    return fvgs


# ══════════════════════════════════════════════════════════════════════════════
# OPTION BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Normal CDF using math.erf — no scipy needed."""
    import math
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def black_scholes_price(S, K, T, r, sigma, option_type="CE"):
    """Black-Scholes using pure math — no scipy."""
    import math
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(S - K, 0) if option_type in ("CE", "CALL") else max(K - S, 0)
        return max(intrinsic, 0.05)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type in ("CE", "CALL"):
            return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    except Exception:
        return 0.05


def simulate_option_prices(df: pd.DataFrame, strike_offset: int,
                            strike_gap: int, opt_type: str,
                            dte: int = 7, iv: float = 0.15) -> pd.DataFrame:
    """
    Simulate option prices for each bar using Black-Scholes.
    strike_offset: integer offset from ATM (e.g. 0=ATM, 1=ATM+1, -1=ATM-1)
    """
    df = df.copy()
    r  = 0.065  # RBI repo rate approx
    prices = []

    for i, row in df.iterrows():
        S = row["close"]
        remaining_dte = max(dte - i * (dte / len(df)), 0.01)
        T = remaining_dte / 365
        K = compute_strike_price(S, strike_gap, strike_offset)
        price = black_scholes_price(S, K, T, r, iv, opt_type)
        prices.append({"strike": K, "opt_price": round(price, 2)})

    opt_df = pd.DataFrame(prices, index=df.index)
    df["strike"]    = opt_df["strike"]
    df["opt_price"] = opt_df["opt_price"]
    return df


def generate_signals(df: pd.DataFrame, buy_lvl: float,
                      sell_lvl: float) -> pd.DataFrame:
    df = df.copy()
    lc = (df["bsp"] > buy_lvl)  & (df["close"] > df["ema20"]) & (df["ema20"] > df["ema50"])
    ec = (df["bsp"] < sell_lvl) & (df["close"] < df["ema20"]) & (df["ema20"] < df["ema50"])
    df["signal"] = np.where(lc, 1, np.where(ec, -1, 0))
    return df


def run_backtest(df: pd.DataFrame, capital: float, size_pct: float,
                 comm_pct: float, lot_size: int,
                 trade_options: bool = False) -> tuple[pd.DataFrame, list]:
    """
    Backtest on index price or on option premium.
    trade_options=True → uses opt_price column for P&L
    """
    cash, pos, entry_price, entry_time = capital, 0, 0.0, None
    in_trade = False
    equities, trades = [], []

    price_col = "opt_price" if (trade_options and "opt_price" in df.columns) else "close"

    for _, row in df.iterrows():
        sig   = row["signal"]
        price = row[price_col]
        ts    = row["timestamp"]

        if sig == 1 and not in_trade:
            invest = cash * size_pct
            lots   = max(int(invest / (price * lot_size)), 1)
            qty    = lots * lot_size
            cost   = qty * price * (1 + comm_pct)
            if cost <= cash:
                pos         = qty
                cash       -= cost
                entry_price = price
                entry_time  = ts
                in_trade    = True

        elif sig == -1 and in_trade:
            proceeds = pos * price * (1 - comm_pct)
            pnl      = proceeds - pos * entry_price
            cash    += proceeds
            trades.append({
                "entry_time":   entry_time,
                "exit_time":    ts,
                "entry_price":  round(entry_price, 2),
                "exit_price":   round(price, 2),
                "qty":          pos,
                "lots":         pos // lot_size,
                "pnl":          round(pnl, 2),
                "return_pct":   round((price / entry_price - 1) * 100, 3),
                "exit_reason":  "Signal",
                "strike":       row.get("strike", "-"),
            })
            pos, in_trade = 0, False

        equities.append(cash + pos * price)

    # Close open trade
    if in_trade and pos > 0:
        lp       = df[price_col].iloc[-1]
        proceeds = pos * lp * (1 - comm_pct)
        pnl      = proceeds - pos * entry_price
        cash    += proceeds
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   df["timestamp"].iloc[-1],
            "entry_price": round(entry_price, 2),
            "exit_price":  round(lp, 2),
            "qty":         pos,
            "lots":        pos // lot_size,
            "pnl":         round(pnl, 2),
            "return_pct":  round((lp / entry_price - 1) * 100, 3),
            "exit_reason": "End of Data",
            "strike":      df["strike"].iloc[-1] if "strike" in df.columns else "-",
        })

    results = df[["timestamp"]].copy()
    results["equity"]       = equities
    results["peak"]         = results["equity"].cummax()
    results["drawdown_pct"] = (results["equity"] - results["peak"]) / results["peak"] * 100
    return results, trades


def metrics(results: pd.DataFrame, trades: list, capital: float) -> dict:
    final     = results["equity"].iloc[-1]
    pnl       = final - capital
    ret       = pnl / capital * 100
    n_days    = (results["timestamp"].iloc[-1] - results["timestamp"].iloc[0]).days
    cagr      = ((final / capital) ** (365 / max(n_days, 1)) - 1) * 100
    max_dd    = results["drawdown_pct"].min()
    rets      = results["equity"].pct_change().dropna()
    sharpe    = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    neg       = rets[rets < 0]
    sortino   = rets.mean() / neg.std() * np.sqrt(252) if len(neg) > 1 and neg.std() > 0 else 0
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0
    pnls      = [t["pnl"] for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    gp, gl    = sum(wins), abs(sum(losses))
    return {
        "total_return": ret,  "total_pnl": pnl,      "cagr": cagr,
        "max_drawdown": max_dd, "sharpe": sharpe,    "sortino": sortino,
        "calmar": calmar,     "total_trades": len(trades),
        "win_rate":  len(wins) / len(trades) * 100 if trades else 0,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_win":  np.mean(wins)   if wins   else 0,
        "avg_loss": np.mean(losses) if losses else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Token Status ─────────────────────────────────────────────────────────
    cfg = DhanConfig()
    if cfg.access_token and cfg.access_token != "paste_your_token_here":
        st.success(f"✅ Token configured · Client: `{cfg.client_id}`")
    else:
        st.error("❌ Token not set — update `access_token` in DhanConfig at top of app.py")

    # ── Index Selection ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">📊 Index Settings</div>', unsafe_allow_html=True)

    index_name = st.selectbox("Select Index", list(INDICES.keys()), index=0)
    idx_cfg    = INDICES[index_name]
    lot_size   = idx_cfg["lot_size"]
    strike_gap = idx_cfg["strike_gap"]

    st.caption(f"Lot Size: **{lot_size}** | Strike Gap: **{strike_gap}**")

    interval_map = {
        "1 Min":"1", "5 Min":"5", "15 Min":"15",
        "25 Min":"25", "60 Min":"60", "Daily":"D"
    }
    interval_lbl = st.selectbox("Timeframe", list(interval_map.keys()), index=3)
    interval     = interval_map[interval_lbl]

    c1, c2 = st.columns(2)
    with c1: from_date = st.date_input("From", datetime.now() - timedelta(days=365))
    with c2: to_date   = st.date_input("To",   datetime.now())

    # ── Backtest Mode ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">🎯 Backtest Mode</div>', unsafe_allow_html=True)

    backtest_mode = st.radio(
        "Trade on",
        ["Index (Futures-style)", "Options (Real Data)"],
        index=0,
        help="Index mode: signals on index OHLCV. Options mode: uses Dhan rollingoption API for real premium data."
    )
    trade_options = backtest_mode == "Options (Real Data)"

    if trade_options:
        st.markdown('<div class="sidebar-header">📌 Strike Selection</div>', unsafe_allow_html=True)

        opt_type = st.radio("Option Type", ["CE (Call)", "PE (Put)"], horizontal=True)
        opt_type = "CALL" if "CE" in opt_type else "PUT"

        # Strike offset selector: ATM-3 … ATM … ATM+3
        offsets       = list(range(-3, 4))
        offset_labels = [get_strike_label(o) for o in offsets]
        sel_label     = st.select_slider(
            "Strike",
            options=offset_labels,
            value="ATM",
            help="ATM = At the Money. +1 = one strike away (OTM for CE). Max ±10 near expiry."
        )
        strike_offset = offsets[offset_labels.index(sel_label)]

        expiry_flag = st.radio("Expiry Type", ["WEEK", "MONTH"], horizontal=True,
                               help="Weekly or Monthly expiry contract")

        st.info(
            f"Fetching **{get_strike_label(strike_offset)} {opt_type}** real option data\n"
            f"via Dhan `/charts/rollingoption`"
        )

    # ── Strategy Parameters ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">⚙️ Strategy Parameters</div>', unsafe_allow_html=True)

    pivot_length = st.slider("Pivot Lookback",  3, 20, 5)
    bsp_length   = st.slider("BSP Length",      5, 50, 21)
    bsp_buy_lvl  = st.number_input("BSP Buy Level",  value=0.08,  step=0.01, format="%.2f")
    bsp_sell_lvl = st.number_input("BSP Sell Level", value=-0.08, step=0.01, format="%.2f")
    vol_threshold = st.slider("Volume Threshold",   1.0, 3.0, 1.5, 0.1)
    vol_period    = st.slider("Volume SMA Period",   5,  30,  14)

    st.markdown('<div class="sidebar-header">📈 EMA Filter</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        show_e20  = st.checkbox("EMA 20",  True)
        show_e50  = st.checkbox("EMA 50",  True)
    with c2:
        show_e100 = st.checkbox("EMA 100", True)
        show_e200 = st.checkbox("EMA 200", True)

    st.markdown('<div class="sidebar-header">💼 Capital</div>', unsafe_allow_html=True)
    init_capital = st.number_input("Initial Capital (₹)", value=500000, step=50000)
    pos_size_pct = st.slider("Position Size (%)", 5, 100, 50)
    comm_pct     = st.number_input("Commission (%)", value=0.03, step=0.01, format="%.3f")

    st.markdown('<div class="sidebar-header">🛠️ Debug</div>', unsafe_allow_html=True)
    debug_mode = st.checkbox("Show API request/response", False,
                              help="Shows raw payloads and responses to diagnose API errors")

    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"## 📡 NYZTrade · SMC Liquidity Lens — Index Backtest")
st.caption("Smart Money Concepts + BSP Oscillator + FVG | Indian Indices | Dhan API")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    # ── 1. Fetch Index OHLCV ─────────────────────────────────────────────────
    with st.spinner(f"📡 Fetching {index_name} OHLCV from Dhan…"):
        df = _fetcher.fetch_index_ohlcv(
            idx_cfg["security_id"],
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
            interval,
            debug=debug_mode
        )

    if df is None or df.empty:
        st.error("❌ No index data returned. Try: shorter date range, Daily timeframe, or refresh your token.")
        st.stop()

    st.success(f"✅ Loaded **{len(df):,} bars** for **{index_name}** ({interval_lbl})")

    # ── 2. Indicators & Signals ───────────────────────────────────────────────
    with st.spinner("⚙️ Computing indicators…"):
        df["ema20"]   = ema(df["close"], 20)
        df["ema50"]   = ema(df["close"], 50)
        df["ema100"]  = ema(df["close"], 100)
        df["ema200"]  = ema(df["close"], 200)
        df["vol_sma"] = df["volume"].rolling(vol_period).mean()
        df["bsp"]     = calc_bsp(df, bsp_length)

        ph   = pivot_highs(df, pivot_length)
        pl   = pivot_lows(df, pivot_length)
        obs  = order_blocks(df, ph, pl, vol_threshold)
        fvgs = fair_value_gaps(df)
        df   = generate_signals(df, bsp_buy_lvl, bsp_sell_lvl)

    # ── 3. Fetch Real Option Data (if options mode) ───────────────────────────
    opt_df = None
    if trade_options:
        with st.spinner(f"📡 Fetching {get_strike_label(strike_offset)} {opt_type} option data…"):
            opt_df = _fetcher.fetch_rolling_option(
                idx_cfg["security_id"],
                strike_offset, opt_type,
                from_date.strftime("%Y-%m-%d"),
                to_date.strftime("%Y-%m-%d"),
                interval=interval if interval != "D" else "25",
                expiry_flag=expiry_flag,
                debug=debug_mode
            )

        if opt_df is not None and not opt_df.empty:
            st.success(f"✅ Option data: **{len(opt_df):,} bars** · {get_strike_label(strike_offset)} {opt_type}")
            # Merge option close price into df on nearest timestamp
            opt_df_r = opt_df[["timestamp","close","oi"]].rename(
                columns={"close":"opt_price","oi":"opt_oi"}
            )
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                opt_df_r.sort_values("timestamp"),
                on="timestamp", direction="nearest", tolerance=pd.Timedelta("30min")
            )
        else:
            st.warning("⚠️ Option data unavailable. Falling back to Black-Scholes simulation.")
            df = simulate_option_prices(df, strike_offset, strike_gap,
                                         opt_type, 7, 0.15)

    # ── 4. Run Backtest ───────────────────────────────────────────────────────
    with st.spinner("📊 Running backtest…"):
        results, trades = run_backtest(
            df, init_capital, pos_size_pct / 100,
            comm_pct / 100, lot_size, trade_options
        )
        m = metrics(results, trades, init_capital)

    # ── 5. Mode Banner ────────────────────────────────────────────────────────
    if trade_options:
        st.info(
            f"📌 **Options Mode (Real Data)** | {index_name} | "
            f"**{get_strike_label(strike_offset)} {opt_type}** | {expiry_flag} expiry"
        )
    else:
        st.info(f"📈 **Index Mode** | {index_name} | Lot Size: {lot_size}")

    # ── 4. KPIs ───────────────────────────────────────────────────────────────
    st.markdown("### 📊 Performance")
    k = st.columns(6)
    k[0].metric("Total Return",  f"{m['total_return']:.1f}%",  delta=f"₹{m['total_pnl']:,.0f}")
    k[1].metric("Profit Factor", f"{m['profit_factor']:.2f}")
    k[2].metric("Win Rate",      f"{m['win_rate']:.1f}%")
    k[3].metric("Max Drawdown",  f"{m['max_drawdown']:.1f}%",  delta_color="inverse")
    k[4].metric("Total Trades",  str(m['total_trades']))
    k[5].metric("Sharpe",        f"{m['sharpe']:.2f}")
    st.divider()

    # ── 5. Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "📋 Trades", "📉 Equity & DD", "📑 Stats"])

    # ─ Chart ──────────────────────────────────────────────────────────────────
    with tab1:
        rows   = 4 if trade_options else 3
        h_rows = [0.50, 0.15, 0.17, 0.18] if trade_options else [0.60, 0.20, 0.20]
        subs   = (["Price + SMC Levels", "Volume", "BSP Oscillator", f"Option Premium ({get_strike_label(strike_offset)} {opt_type})"]
                  if trade_options else
                  ["Price + SMC Levels", "Volume", "BSP Oscillator"])

        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            row_heights=h_rows, vertical_spacing=0.025,
            subplot_titles=subs
        )

        # Candles
        fig.add_trace(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name=index_name,
            increasing_line_color="#00d4aa", decreasing_line_color="#ff4b6e"
        ), row=1, col=1)

        # EMAs
        for show, col, name, color in [
            (show_e20,  "ema20",  "EMA 20",  "#ff4b6e"),
            (show_e50,  "ema50",  "EMA 50",  "#ff9900"),
            (show_e100, "ema100", "EMA 100", "#00bcd4"),
            (show_e200, "ema200", "EMA 200", "#3f8ef5"),
        ]:
            if show:
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=name,
                                          line=dict(color=color, width=1),
                                          opacity=0.85), row=1, col=1)

        # Order Blocks
        for ob in obs[:60]:
            fc = "rgba(255,75,110,0.10)" if ob["type"] == "bearish" else "rgba(0,212,170,0.10)"
            bc = "#ff4b6e"              if ob["type"] == "bearish" else "#00d4aa"
            x0 = df["timestamp"].iloc[ob["start"]]
            x1 = df["timestamp"].iloc[min(ob["end"], len(df) - 1)]
            fig.add_hrect(y0=ob["btm"], y1=ob["top"], x0=x0, x1=x1,
                          fillcolor=fc, line_color=bc, line_width=0.5,
                          annotation_text="OB★" if ob["strong"] else "OB",
                          annotation_font_color=bc, annotation_font_size=8,
                          row=1, col=1)

        # FVGs
        for fvg in fvgs[:40]:
            x0 = df["timestamp"].iloc[fvg["start"]]
            x1 = df["timestamp"].iloc[min(fvg["start"] + 15, len(df) - 1)]
            fig.add_hrect(y0=fvg["btm"], y1=fvg["top"], x0=x0, x1=x1,
                          fillcolor="rgba(150,0,255,0.09)",
                          line_color="#9000ff", line_width=0.5,
                          annotation_text="FVG",
                          annotation_font_color="#b060ff", annotation_font_size=8,
                          row=1, col=1)

        # Signals
        buys  = df[df["signal"] ==  1]
        sells = df[df["signal"] == -1]
        fig.add_trace(go.Scatter(x=buys["timestamp"],  y=buys["low"]  * 0.998,
                                  mode="markers", name="BUY",
                                  marker=dict(symbol="triangle-up",   size=9, color="#00d4aa")), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"] * 1.002,
                                  mode="markers", name="EXIT",
                                  marker=dict(symbol="triangle-down", size=9, color="#ff4b6e")), row=1, col=1)

        # Volume
        vcol = ["#00d4aa" if c >= o else "#ff4b6e"
                for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"],
                              name="Volume", marker_color=vcol, opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vol_sma"],
                                  name="Vol SMA", line=dict(color="#ff9900", width=1)), row=2, col=1)

        # BSP
        bsp_col = ["#00d4aa" if v > 0 else "#ff4b6e" for v in df["bsp"].fillna(0)]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["bsp"],
                              name="BSP", marker_color=bsp_col, opacity=0.85), row=3, col=1)
        fig.add_hline(y=bsp_buy_lvl,  line_color="#00d4aa", line_dash="dash", line_width=1, row=3, col=1)
        fig.add_hline(y=bsp_sell_lvl, line_color="#ff4b6e", line_dash="dash", line_width=1, row=3, col=1)
        fig.add_hline(y=0,            line_color="#444",    line_width=1,      row=3, col=1)

        # Option Premium panel
        if trade_options and "opt_price" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["opt_price"],
                name=f"{get_strike_label(strike_offset)} {opt_type}",
                line=dict(color="#d966ff", width=1.5), fill="tozeroy",
                fillcolor="rgba(150,0,255,0.08)"
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=buys["timestamp"],  y=buys["opt_price"],
                mode="markers", name="Buy Option",
                marker=dict(symbol="triangle-up", size=9, color="#00d4aa")
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=sells["timestamp"], y=sells["opt_price"],
                mode="markers", name="Exit Option",
                marker=dict(symbol="triangle-down", size=9, color="#ff4b6e")
            ), row=4, col=1)

        fig.update_layout(
            height=900, template="plotly_dark", showlegend=True,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            margin=dict(t=35, b=20)
        )
        fig.update_xaxes(showgrid=True, gridcolor="#1e2130")
        fig.update_yaxes(showgrid=True, gridcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)

    # ─ Trades ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Trade Log")
        if trades:
            tdf     = pd.DataFrame(trades)
            cols    = ["entry_time","exit_time","entry_price","exit_price",
                       "qty","lots","strike","pnl","return_pct","exit_reason"]
            cols    = [c for c in cols if c in tdf.columns]
            styled  = (
                tdf[cols]
                .rename(columns={
                    "entry_time":"Entry", "exit_time":"Exit",
                    "entry_price":"Entry ₹","exit_price":"Exit ₹",
                    "qty":"Qty","lots":"Lots","strike":"Strike",
                    "pnl":"P&L ₹","return_pct":"Return %","exit_reason":"Reason"
                })
                .style.applymap(
                    lambda v: "color:#00d4aa" if isinstance(v, (int,float)) and v > 0
                              else ("color:#ff4b6e" if isinstance(v, (int,float)) and v < 0 else ""),
                    subset=["P&L ₹","Return %"]
                )
            )
            st.dataframe(styled, use_container_width=True, height=500)
            csv = pd.DataFrame(trades).to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv,
                               f"{index_name}_trades.csv", "text/csv")
        else:
            st.info("No trades generated. Try adjusting BSP levels or timeframe.")

    # ─ Equity & Drawdown ──────────────────────────────────────────────────────
    with tab3:
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.65, 0.35], vertical_spacing=0.05,
                              subplot_titles=("Equity Curve (₹)", "Drawdown (%)"))
        fig2.add_trace(go.Scatter(
            x=results["timestamp"], y=results["equity"],
            name="Portfolio", fill="tozeroy",
            line=dict(color="#00d4aa", width=2),
            fillcolor="rgba(0,212,170,0.08)"
        ), row=1, col=1)
        fig2.add_hline(y=init_capital, line_color="#555", line_dash="dash", row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=results["timestamp"], y=results["drawdown_pct"],
            name="Drawdown", fill="tozeroy",
            line=dict(color="#ff4b6e", width=1),
            fillcolor="rgba(255,75,110,0.12)"
        ), row=2, col=1)
        fig2.update_layout(
            height=500, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(t=35, b=20)
        )
        fig2.update_xaxes(showgrid=True, gridcolor="#1e2130")
        fig2.update_yaxes(showgrid=True, gridcolor="#1e2130")
        st.plotly_chart(fig2, use_container_width=True)

    # ─ Stats ──────────────────────────────────────────────────────────────────
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Return Metrics**")
            for k2, v in [
                ("Total Return (%)",  f"{m['total_return']:.2f}%"),
                ("Total P&L (₹)",     f"₹{m['total_pnl']:,.2f}"),
                ("CAGR (%)",          f"{m['cagr']:.2f}%"),
                ("Sharpe Ratio",      f"{m['sharpe']:.3f}"),
                ("Sortino Ratio",     f"{m['sortino']:.3f}"),
                ("Calmar Ratio",      f"{m['calmar']:.3f}"),
            ]:
                st.markdown(f"`{k2}` &nbsp; **{v}**")
        with c2:
            st.markdown("**Trade Metrics**")
            for k2, v in [
                ("Total Trades",     m["total_trades"]),
                ("Win Rate (%)",     f"{m['win_rate']:.1f}%"),
                ("Profit Factor",    f"{m['profit_factor']:.3f}"),
                ("Avg Win (₹)",      f"₹{m['avg_win']:,.2f}"),
                ("Avg Loss (₹)",     f"₹{m['avg_loss']:,.2f}"),
                ("Max Drawdown (%)", f"{m['max_drawdown']:.2f}%"),
            ]:
                st.markdown(f"`{k2}` &nbsp; **{v}**")

        if len(results) > 30:
            st.markdown("#### Monthly Returns")
            r2 = results.copy()
            r2["month"] = r2["timestamp"].dt.to_period("M")
            monthly = r2.groupby("month")["equity"].last().pct_change() * 100
            monthly.index = monthly.index.astype(str)
            monthly = monthly.dropna()
            fig3 = go.Figure(go.Bar(
                x=monthly.index, y=monthly.values,
                marker_color=["#00d4aa" if v >= 0 else "#ff4b6e" for v in monthly.values],
                text=[f"{v:.1f}%" for v in monthly.values],
                textposition="outside", textfont=dict(size=9)
            ))
            fig3.update_layout(
                height=280, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                xaxis=dict(tickangle=-45), margin=dict(t=10, b=50),
                yaxis_title="Return (%)"
            )
            st.plotly_chart(fig3, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; opacity:0.65;">
        <h2>📡 SMC Liquidity Lens — Index Backtester</h2>
        <p style="font-size:1.05rem;">
            Paste your <b>Dhan Access Token</b>, select an <b>Index</b> and <b>Strike</b>,
            then click <b>Run Backtest</b>.
        </p>
        <br>
        <table style="margin:auto; font-size:0.9rem; border-collapse:collapse; line-height:2.2;">
            <tr>
                <td style="padding:4px 20px;">✅ NIFTY · BANKNIFTY · FINNIFTY · MIDCPNIFTY</td>
                <td style="padding:4px 20px;">✅ SENSEX · BANKEX</td>
            </tr>
            <tr>
                <td style="padding:4px 20px;">✅ Index Futures-style Backtest</td>
                <td style="padding:4px 20px;">✅ Options: ATM-3 → ATM → ATM+3</td>
            </tr>
            <tr>
                <td style="padding:4px 20px;">✅ BSP + SMC Order Blocks + FVG</td>
                <td style="padding:4px 20px;">✅ Sharpe · Sortino · Calmar · CAGR</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
