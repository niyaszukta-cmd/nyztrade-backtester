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
    access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcyMzQ5MzA1LCJhcHBfaWQiOiJjOTNkM2UwOSIsImlhdCI6MTc3MjI2MjkwNSwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.2Gx9EDLxt0avGLTwbu4zriOX03VIwQMAF2xHmC9NzEq6jEkSgMSpGTLNbpOh2ENEU3Rd6TrD5Fcmvsm3Ca0Xkg"   # ← paste fresh token here daily

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
        # Dhan natively supports: 1, 5, 15, 25, 60
        # 3-min and 10-min are fetched as 1-min and resampled client-side
        RESAMPLE_MAP = {"3": ("1", 3), "10": ("1", 10)}
        resample_to = None
        fetch_interval = interval
        if interval in RESAMPLE_MAP:
            fetch_interval, resample_to = RESAMPLE_MAP[interval]

        intraday_set = {"1", "5", "15", "25", "60"}

        if fetch_interval in intraday_set:
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
    """
    BSP (Balance of Selling Pressure) — exact Pine Script translation.
    Pine: ad = (close==high and close==low) or high==low ? 0 : ((2*close-low-high)/(high-low))*volume
    mf  = sum(ad, length) / sum(volume, length)
    """
    hl = df["high"] - df["low"]
    flat = (hl == 0) | ((df["close"] == df["high"]) & (df["close"] == df["low"]))
    ad   = np.where(flat, 0.0, ((2 * df["close"] - df["low"] - df["high"]) / hl.replace(0, 1.0)) * df["volume"])
    ad   = pd.Series(ad, index=df.index)
    vol_sum = df["volume"].rolling(length).sum()
    return (ad.rolling(length).sum() / vol_sum).fillna(0)


def calc_bsp_oiv(df: pd.DataFrame, length: int) -> pd.Series:
    """
    BSP variant for option strike charts — uses OI × Volume as the weight
    instead of plain volume. This amplifies commitment signal:
    high OI + high volume = smart money repositioning.

    Formula: ad = ((2*close - low - high) / (high - low)) * (oi * volume)
             bsp = sum(ad, length) / sum(oi*volume, length)

    Falls back to plain volume if OI column is missing or all-NaN.
    """
    if "oi" not in df.columns or df["oi"].isna().all():
        return calc_bsp(df, length)  # fallback

    oi_vol = (df["oi"].fillna(0) * df["volume"].fillna(0))
    hl   = df["high"] - df["low"]
    flat = (hl == 0) | ((df["close"] == df["high"]) & (df["close"] == df["low"]))
    ad   = np.where(flat, 0.0,
                    ((2 * df["close"] - df["low"] - df["high"]) / hl.replace(0, 1.0)) * oi_vol)
    ad      = pd.Series(ad, index=df.index)
    oiv_sum = oi_vol.rolling(length).sum()
    return (ad.rolling(length).sum() / oiv_sum.replace(0, np.nan)).fillna(0)


def calc_bsp_daily(df_intraday: pd.DataFrame, daily_df: pd.DataFrame, length: int) -> pd.Series:
    """
    Pine uses request.security(syminfo.tickerid, 'D', mf) — BSP is computed on
    DAILY bars and then stamp-forwarded onto each intraday bar (same value all day).
    This function replicates that: compute BSP on daily_df, then merge onto intraday.
    """
    if daily_df is None or daily_df.empty:
        return calc_bsp(df_intraday, length)  # fallback to intraday BSP
    # Compute on daily bars
    hl = daily_df["high"] - daily_df["low"]
    flat = (hl == 0) | ((daily_df["close"] == daily_df["high"]) & (daily_df["close"] == daily_df["low"]))
    ad   = np.where(flat, 0.0,
                    ((2 * daily_df["close"] - daily_df["low"] - daily_df["high"])
                     / hl.replace(0, 1.0)) * daily_df["volume"])
    ad   = pd.Series(ad, index=daily_df.index)
    vol_sum = daily_df["volume"].rolling(length).sum()
    daily_bsp = (ad.rolling(length).sum() / vol_sum).fillna(0)
    daily_bsp.index = pd.to_datetime(daily_df["timestamp"].values).normalize()
    # Stamp onto intraday bars by date
    intra_dates = pd.to_datetime(df_intraday["timestamp"].values).normalize()
    return pd.Series(intra_dates, index=df_intraday.index).map(
        daily_bsp.to_dict()
    ).ffill().fillna(0)


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


def generate_signals(df: pd.DataFrame, buy_lvl: float, sell_lvl: float,
                      ema_filter: bool = True, signal_mode: str = "Pine Exact") -> pd.DataFrame:
    """
    Signal modes:

    Pine Exact  ← DEFAULT — exact replica of TradingView Pine Script logic:
                  ENTRY: bsp > buyLevel AND close > ema20 AND ema20 > ema50
                  EXIT:  bsp < sellLevel AND close < ema20 AND ema20 < ema50
                  (strategy.entry BUY + strategy.close BUY — no shorts taken)

    Level Hold  — entry while BSP sustained above level; exit at neutral BSP
                  (more trades on long historical data)

    Flip        — entry only on BSP level CROSS; exit only on cross below sell level

    BSP Only    — no EMA filter, pure BSP threshold (most trades)
    """
    df = df.copy()

    # EMA trend conditions — shared by all modes
    has_ema = "ema20" in df.columns and "ema50" in df.columns
    if has_ema and ema_filter:
        bull_trend = (df["close"] > df["ema20"]) & (df["ema20"] > df["ema50"])
        bear_trend = (df["close"] < df["ema20"]) & (df["ema20"] < df["ema50"])
    else:
        bull_trend = pd.Series(True, index=df.index)
        bear_trend = pd.Series(True, index=df.index)

    bsp = df["bsp"]

    if signal_mode == "Pine Exact":
        # ── EXACT Pine Script replication ────────────────────────────────────
        # longCondition  = bsp > bspBuyLevel  AND close > ema20 AND ema20 > ema50
        # shortCondition = bsp < bspSellLevel AND close < ema20 AND ema20 < ema50
        # strategy.entry("BUY", strategy.long) when longCondition
        # strategy.close("BUY")               when shortCondition
        long_entry  = (bsp > buy_lvl)  & bull_trend
        exit_signal = (bsp < sell_lvl) & bear_trend
        df["signal"] = np.where(long_entry, 1, np.where(exit_signal, -1, 0))
        return df

    elif signal_mode == "Level Hold":
        long_entry  = (bsp > buy_lvl)  & bull_trend
        short_entry = (bsp < sell_lvl) & bear_trend
        long_exit   = bsp < 0
        short_exit  = bsp > 0
        sigs = [0] * len(df)
        for i in range(1, len(df)):
            prev = sigs[i - 1]
            if long_entry.iloc[i]:
                sigs[i] = 1
            elif short_entry.iloc[i]:
                sigs[i] = -1
            elif prev == 1 and (long_exit.iloc[i] or short_entry.iloc[i]):
                sigs[i] = -1
            elif prev == -1 and (short_exit.iloc[i] or long_entry.iloc[i]):
                sigs[i] = 1
            else:
                sigs[i] = 0
        df["signal"] = sigs
        return df

    elif signal_mode == "BSP Only":
        long_entry  = bsp > buy_lvl
        short_entry = bsp < sell_lvl

    else:  # Flip — BSP level cross
        long_entry  = (bsp > buy_lvl)  & (bsp.shift(1) <= buy_lvl)  & bull_trend
        short_entry = (bsp < sell_lvl) & (bsp.shift(1) >= sell_lvl) & bear_trend

    df["signal"] = np.where(long_entry, 1, np.where(short_entry, -1, 0))
    return df


def run_backtest(df: pd.DataFrame, capital: float, size_pct: float,
                 comm_pct: float, lot_size: int,
                 trade_options: bool = False,
                 fixed_lots: int = None,
                 spread_legs: list = None,
                 is_intraday: bool = False,
                 eod_exit_time=None) -> tuple[pd.DataFrame, list]:
    """
    Spread-aware lot-based backtest.

    Single leg:  price_col = opt_price (or close for index mode)
    Multi-leg:   spread_price column = sum of signed leg prices already on df.
                 Per-leg P&L breakdown recorded in each trade dict.

    Sizing:
      fixed_lots → that many SCALE lots (each leg uses its own leg["lots"] ratio)
      % of Capital → lots = floor(budget / |spread_cost_per_lot|)

    is_intraday: force-close all positions at eod_exit_time each trading day
    eod_exit_time: datetime.time object for EOD square-off (default 15:15)
    """
    cash, pos_lots, entry_spread, entry_time = capital, 0, 0.0, None
    in_trade = False
    equities, trades = [], []

    # Determine price column
    if trade_options and spread_legs and "spread_price" in df.columns:
        price_col = "spread_price"
        is_spread  = True
    elif trade_options and "opt_price" in df.columns:
        price_col = "opt_price"
        is_spread  = False
    else:
        price_col = "close"
        is_spread  = False

    # Build per-leg price lookups for spread mode
    leg_price_series = {}
    if is_spread and spread_legs:
        ts_index = df["timestamp"].values
        for i, ld in enumerate(spread_legs):
            leg_price_series[i] = ld["prices"].reindex(
                pd.Index(ts_index)
            ).ffill().fillna(0).values

    df_ts = df["timestamp"].values  # fast numpy access

    import datetime as _dt
    # Resolve EOD exit time
    if is_intraday and eod_exit_time is not None:
        _eod_h, _eod_m = eod_exit_time.hour, eod_exit_time.minute
    else:
        _eod_h, _eod_m = 15, 15   # default fallback

    for row_i, row in df.iterrows():
        sig = row["signal"]
        ts  = row["timestamp"]
        raw_price = row[price_col]

        if pd.isna(raw_price):
            equities.append(cash + pos_lots * (entry_spread if in_trade else 0))
            continue

        # Spread price = net debit/credit per scale-lot
        spread_price = float(raw_price)

        # ── INTRADAY EOD FORCE-CLOSE ──────────────────────────────────────
        # If Intraday mode: square off at/after eod_exit_time on each day
        if is_intraday and in_trade:
            bar_time = ts.time() if hasattr(ts, "time") else (
                _dt.datetime.fromtimestamp(ts.timestamp()).time()
                if hasattr(ts, "timestamp") else None
            )
            if bar_time and bar_time >= _dt.time(_eod_h, _eod_m):
                eod_spread = spread_price
                qty        = pos_lots * lot_size
                pnl        = (eod_spread - entry_spread) * qty - abs(eod_spread) * qty * comm_pct
                cash      += abs(entry_spread) * qty + pnl
                trade_rec  = {
                    "entry_time":  entry_time, "exit_time": ts,
                    "entry_price": round(entry_spread, 2), "exit_price": round(eod_spread, 2),
                    "qty": qty, "lots": pos_lots,
                    "pnl": round(pnl, 2),
                    "return_pct": round((eod_spread / entry_spread - 1) * 100, 2) if entry_spread != 0 else 0.0,
                    "exit_reason": "EOD Square-off",
                    "strike": row.get("strike", "-"),
                }
                if is_spread and spread_legs:
                    for i, ld in enumerate(spread_legs):
                        lp_arr = leg_price_series.get(i)
                        if lp_arr is not None and row_i < len(lp_arr):
                            try:
                                eidx = list(df_ts).index(entry_time)
                                ep = float(lp_arr[eidx]); xp = float(lp_arr[row_i])
                            except (ValueError, IndexError):
                                ep = xp = 0.0
                            sign = 1 if ld["direction"] == "BUY" else -1
                            trade_rec[f"leg{i+1}_{ld['strike_lbl']}_{ld['opt_type']}"] = round(
                                sign * (xp - ep) * ld["lots"] * lot_size, 2)
                trades.append(trade_rec)
                pos_lots, in_trade = 0, False
                equities.append(cash)
                continue  # skip further processing for this bar

        # ── ENTRY — trigger on BUY signal (or SELL signal for short legs) ──
        # Entry fires on sig==1 (long) or sig==-1 when no position open
        # In Intraday mode: block new entries after EOD time
        _bar_time_now = ts.time() if hasattr(ts, "time") else None
        _block_entry  = is_intraday and _bar_time_now and _bar_time_now >= _dt.time(_eod_h, _eod_m)
        entry_triggered = (sig == 1 and not in_trade and not _block_entry)

        if entry_triggered:
            # Cost basis: |spread_price| × lot_size per scale-lot
            cost_per_lot = abs(spread_price) * lot_size
            if cost_per_lot < 0.01:
                equities.append(cash)
                continue

            if fixed_lots is not None:
                scale_lots = int(fixed_lots)
            else:
                scale_lots = max(int(cash * size_pct / cost_per_lot), 1)

            total_cost = cost_per_lot * scale_lots * (1 + comm_pct)
            if total_cost <= cash:
                pos_lots     = scale_lots
                cash        -= total_cost
                entry_spread = spread_price
                entry_time   = ts
                in_trade     = True

        # ── EXIT — on sell signal OR new buy when already in trade ─────────
        elif in_trade and sig == -1:
            exit_spread = spread_price
            qty         = pos_lots * lot_size  # total underlying units

            gross_pnl = (exit_spread - entry_spread) * qty
            comm_cost = abs(exit_spread) * qty * comm_pct
            pnl       = gross_pnl - comm_cost
            cash     += abs(entry_spread) * qty + pnl  # return capital + P&L

            trade_rec = {
                "entry_time":    entry_time,
                "exit_time":     ts,
                "entry_price":   round(entry_spread, 2),
                "exit_price":    round(exit_spread,  2),
                "qty":           qty,
                "lots":          pos_lots,
                "pnl":           round(pnl, 2),
                "return_pct":    round((exit_spread / entry_spread - 1) * 100, 2)
                                 if entry_spread != 0 else 0.0,
                "exit_reason":   "Signal",
                "strike":        row.get("strike", "-"),
            }

            # Per-leg P&L breakdown for spread trades
            if is_spread and spread_legs:
                for i, ld in enumerate(spread_legs):
                    lp_arr  = leg_price_series.get(i)
                    if lp_arr is not None and row_i < len(lp_arr):
                        # Find entry row index (approx by timestamp match)
                        try:
                            entry_idx = list(df_ts).index(entry_time)
                            ep = float(lp_arr[entry_idx])
                            xp = float(lp_arr[row_i])
                        except (ValueError, IndexError):
                            ep = xp = 0.0
                        sign      = 1 if ld["direction"] == "BUY" else -1
                        leg_pnl   = sign * (xp - ep) * ld["lots"] * lot_size
                        trade_rec[f"leg{i+1}_{ld['strike_lbl']}_{ld['opt_type']}"] = round(leg_pnl, 2)

            trades.append(trade_rec)
            pos_lots, in_trade = 0, False

        # Mark-to-market equity
        mtm = spread_price * pos_lots * lot_size if in_trade else 0
        equities.append(cash + mtm)

    # Force-close open position at last bar
    if in_trade and pos_lots > 0:
        lp = float(df[price_col].iloc[-1])
        qty = pos_lots * lot_size
        pnl = (lp - entry_spread) * qty - abs(lp) * qty * comm_pct
        cash += abs(entry_spread) * qty + pnl
        trade_rec = {
            "entry_time":  entry_time, "exit_time": df["timestamp"].iloc[-1],
            "entry_price": round(entry_spread, 2), "exit_price": round(lp, 2),
            "qty": qty, "lots": pos_lots,
            "pnl": round(pnl, 2),
            "return_pct": round((lp / entry_spread - 1) * 100, 2) if entry_spread != 0 else 0.0,
            "exit_reason": "End of Data",
            "strike": df["strike"].iloc[-1] if "strike" in df.columns else "-",
        }
        if is_spread and spread_legs:
            for i, ld in enumerate(spread_legs):
                lp_arr = leg_price_series.get(i)
                if lp_arr is not None:
                    try:
                        entry_idx = list(df_ts).index(entry_time)
                        ep = float(lp_arr[entry_idx])
                        xp = float(lp_arr[-1])
                    except (ValueError, IndexError):
                        ep = xp = 0.0
                    sign    = 1 if ld["direction"] == "BUY" else -1
                    leg_pnl = sign * (xp - ep) * ld["lots"] * lot_size
                    trade_rec[f"leg{i+1}_{ld['strike_lbl']}_{ld['opt_type']}"] = round(leg_pnl, 2)
        trades.append(trade_rec)
        if equities:
            equities[-1] = cash

    while len(equities) < len(df):
        equities.append(equities[-1] if equities else capital)

    results = df[["timestamp"]].copy()
    results["equity"]       = equities[:len(df)]
    results["peak"]         = results["equity"].cummax()
    results["drawdown_pct"] = (results["equity"] - results["peak"]) / results["peak"] * 100
    return results, trades


def run_backtest_dual(
    ce_df: pd.DataFrame, pe_df: pd.DataFrame,
    capital: float, size_pct: float, comm_pct: float, lot_size: int,
    buy_lvl: float, sell_lvl: float,
    bsp_length: int, signal_mode: str, ema_filter: bool,
    fixed_lots: int = None,
    is_intraday: bool = False,
    eod_exit_time=None
) -> tuple[pd.DataFrame, list]:
    """
    Strike-chart dual backtest.
    - ce_df: ATM CE candles with OI+Volume → BSP computed → CE signals
    - pe_df: ATM PE candles with OI+Volume → BSP computed → PE signals
    - CE signal fires CE BUY trade (buy call when CE OI+Vol momentum is bullish)
    - PE signal fires PE BUY trade (buy put when PE OI+Vol momentum is bullish)
    - Both legs tracked independently; combined equity curve returned.
    """
    import datetime as _dt

    # Compute BSP and signals on each strike df
    ce_df = ce_df.copy()
    pe_df = pe_df.copy()
    ce_df["bsp"] = calc_bsp_oiv(ce_df, bsp_length)
    pe_df["bsp"] = calc_bsp_oiv(pe_df, bsp_length)
    ce_df = generate_signals(ce_df, buy_lvl, sell_lvl, ema_filter=ema_filter, signal_mode=signal_mode)
    pe_df = generate_signals(pe_df, buy_lvl, sell_lvl, ema_filter=ema_filter, signal_mode=signal_mode)

    # Align both dataframes on timestamp
    ce_df = ce_df.set_index("timestamp").sort_index()
    pe_df = pe_df.set_index("timestamp").sort_index()

    all_ts = ce_df.index.union(pe_df.index)
    ce_df  = ce_df.reindex(all_ts).ffill()
    pe_df  = pe_df.reindex(all_ts).ffill()

    if is_intraday and eod_exit_time is not None:
        _eod_h, _eod_m = eod_exit_time.hour, eod_exit_time.minute
    else:
        _eod_h, _eod_m = 15, 15

    cash = capital
    equities, trades = [], []

    # CE leg state
    ce_in_trade = False; ce_lots = 0; ce_entry_price = 0.0; ce_entry_time = None
    # PE leg state
    pe_in_trade = False; pe_lots = 0; pe_entry_price = 0.0; pe_entry_time = None

    def _lots(price):
        if fixed_lots:
            return fixed_lots
        if price <= 0:
            return 0
        budget = cash * size_pct
        return max(1, int(budget / (abs(price) * lot_size)))

    for ts in all_ts:
        ce_row = ce_df.loc[ts] if ts in ce_df.index else None
        pe_row = pe_df.loc[ts] if ts in pe_df.index else None

        ce_sig = int(ce_row["signal"]) if ce_row is not None and pd.notna(ce_row.get("signal")) else 0
        pe_sig = int(pe_row["signal"]) if pe_row is not None and pd.notna(pe_row.get("signal")) else 0
        ce_price = float(ce_row["close"]) if ce_row is not None and pd.notna(ce_row.get("close")) else None
        pe_price = float(pe_row["close"]) if pe_row is not None and pd.notna(pe_row.get("close")) else None

        bar_time = ts.time() if hasattr(ts, "time") else None
        is_eod   = bar_time and bar_time >= _dt.time(_eod_h, _eod_m) if is_intraday else False

        def _exit(in_trade, lots, entry_price, entry_time, price, reason, leg):
            nonlocal cash
            if not in_trade or price is None:
                return False, 0, 0.0, None
            qty = lots * lot_size
            pnl = (price - entry_price) * qty - abs(price) * qty * comm_pct
            cash += abs(entry_price) * qty + pnl
            trades.append({
                "leg": leg,
                "entry_time":  entry_time,  "exit_time": ts,
                "entry_price": round(entry_price, 2), "exit_price": round(price, 2),
                "qty": qty, "lots": lots,
                "pnl": round(pnl, 2),
                "return_pct": round((price / entry_price - 1) * 100, 2) if entry_price != 0 else 0.0,
                "exit_reason": reason,
            })
            return False, 0, 0.0, None

        # ── CE leg ──────────────────────────────────────────────────────────
        if ce_price is not None:
            if is_eod and ce_in_trade:
                ce_in_trade, ce_lots, ce_entry_price, ce_entry_time = _exit(
                    ce_in_trade, ce_lots, ce_entry_price, ce_entry_time, ce_price, "EOD", "CE")
            elif ce_sig == 1 and not ce_in_trade:
                n = _lots(ce_price)
                if n > 0:
                    cost = abs(ce_price) * n * lot_size
                    if cash >= cost:
                        cash -= cost
                        ce_in_trade = True; ce_lots = n
                        ce_entry_price = ce_price; ce_entry_time = ts
            elif ce_sig == -1 and ce_in_trade:
                ce_in_trade, ce_lots, ce_entry_price, ce_entry_time = _exit(
                    ce_in_trade, ce_lots, ce_entry_price, ce_entry_time, ce_price, "Signal", "CE")

        # ── PE leg ──────────────────────────────────────────────────────────
        if pe_price is not None:
            if is_eod and pe_in_trade:
                pe_in_trade, pe_lots, pe_entry_price, pe_entry_time = _exit(
                    pe_in_trade, pe_lots, pe_entry_price, pe_entry_time, pe_price, "EOD", "PE")
            elif pe_sig == 1 and not pe_in_trade:
                n = _lots(pe_price)
                if n > 0:
                    cost = abs(pe_price) * n * lot_size
                    if cash >= cost:
                        cash -= cost
                        pe_in_trade = True; pe_lots = n
                        pe_entry_price = pe_price; pe_entry_time = ts
            elif pe_sig == -1 and pe_in_trade:
                pe_in_trade, pe_lots, pe_entry_price, pe_entry_time = _exit(
                    pe_in_trade, pe_lots, pe_entry_price, pe_entry_time, pe_price, "Signal", "PE")

        # Mark-to-market equity
        mtm = cash
        if ce_in_trade and ce_price:
            mtm += (ce_price - ce_entry_price) * ce_lots * lot_size
        if pe_in_trade and pe_price:
            mtm += (pe_price - pe_entry_price) * pe_lots * lot_size
        equities.append(mtm)

    results = pd.DataFrame({"timestamp": list(all_ts), "equity": equities[:len(all_ts)]})
    results["peak"]         = results["equity"].cummax()
    results["drawdown_pct"] = (results["equity"] - results["peak"]) / results["peak"] * 100
    return results, trades


def run_backtest_alternating(
    df: pd.DataFrame,
    ce_df: pd.DataFrame,
    pe_df: pd.DataFrame,
    capital: float,
    size_pct: float,
    comm_pct: float,
    lot_size: int,
    fixed_lots: int = None,
    is_intraday: bool = False,
    eod_exit_time=None,
) -> tuple[pd.DataFrame, list]:
    """
    CE/PE Alternating State Machine — mirrors Pine Script logic exactly:

        BUY signal  → if PE open: close PE first → open CE
                      if flat:                       open CE
                      if CE open:                    do nothing (no re-entry)

        SELL signal → if CE open: close CE first → open PE
                      if flat:                       open PE
                      if PE open:                    do nothing

    ce_df / pe_df: raw option OHLCV (close = premium price per unit)
    Both are aligned to df index by timestamp.

    Returns (results DataFrame with equity curve, trades list).
    Each trade dict carries 'leg' = 'CE' or 'PE'.
    """
    import datetime as _dt

    # Align CE / PE prices onto main df timestamp
    def _align(opt_raw: pd.DataFrame, name: str) -> pd.Series:
        if opt_raw is None or opt_raw.empty:
            return pd.Series(np.nan, index=df.index)
        tmp = opt_raw[["timestamp", "close"]].rename(columns={"close": name})
        merged = pd.merge_asof(
            df[["timestamp"]].sort_values("timestamp").reset_index(drop=False),
            tmp.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
        ).set_index("index").sort_index()
        return merged[name]

    ce_price_s = _align(ce_df, "ce_price")
    pe_price_s = _align(pe_df, "pe_price")

    if is_intraday and eod_exit_time is not None:
        _eod_h, _eod_m = eod_exit_time.hour, eod_exit_time.minute
    else:
        _eod_h, _eod_m = 15, 15

    def _calc_lots(price: float) -> int:
        if price <= 0:
            return 0
        if fixed_lots is not None:
            return int(fixed_lots)
        return max(1, int(capital * size_pct / (abs(price) * lot_size)))

    cash = capital
    equities, trades = [], []

    # State:  0 = flat, 1 = CE open, -1 = PE open
    active_leg   = 0
    entry_price  = 0.0
    entry_time   = None
    n_lots       = 0

    def _close_position(exit_p: float, exit_ts, reason: str) -> None:
        nonlocal cash, active_leg, entry_price, entry_time, n_lots
        qty  = n_lots * lot_size
        pnl  = (exit_p - entry_price) * qty - abs(exit_p) * qty * comm_pct
        cash += abs(entry_price) * qty + pnl
        trades.append({
            "leg":         "CE" if active_leg == 1 else "PE",
            "entry_time":  entry_time,
            "exit_time":   exit_ts,
            "entry_price": round(entry_price, 2),
            "exit_price":  round(exit_p, 2),
            "qty":         qty,
            "lots":        n_lots,
            "pnl":         round(pnl, 2),
            "return_pct":  round((exit_p / entry_price - 1) * 100, 2) if entry_price else 0.0,
            "exit_reason": reason,
        })
        active_leg  = 0
        entry_price = 0.0
        entry_time  = None
        n_lots      = 0

    def _open_position(leg: int, price: float, ts):
        nonlocal cash, active_leg, entry_price, entry_time, n_lots
        nl   = _calc_lots(price)
        cost = abs(price) * nl * lot_size * (1 + comm_pct)
        if nl > 0 and cash >= cost:
            cash        -= cost
            active_leg   = leg
            entry_price  = price
            entry_time   = ts
            n_lots       = nl

    for row_i, row in df.iterrows():
        sig = row["signal"]
        ts  = row["timestamp"]
        ce_p = ce_price_s.iloc[row_i] if row_i < len(ce_price_s) else np.nan
        pe_p = pe_price_s.iloc[row_i] if row_i < len(pe_price_s) else np.nan

        # ── EOD force-close ───────────────────────────────────────────────────
        if is_intraday and active_leg != 0:
            bar_time = ts.time() if hasattr(ts, "time") else None
            if bar_time and bar_time >= _dt.time(_eod_h, _eod_m):
                eod_p = ce_p if active_leg == 1 else pe_p
                if not np.isnan(eod_p):
                    _close_position(eod_p, ts, "EOD Square-off")
                    equities.append(cash)
                    continue

        # ── BUY signal → target CE ────────────────────────────────────────────
        if sig == 1:
            if active_leg == -1 and not np.isnan(pe_p):
                _close_position(pe_p, ts, "Close PE → Open CE")
            if active_leg == 0 and not np.isnan(ce_p):
                _open_position(1, ce_p, ts)

        # ── SELL signal → target PE ───────────────────────────────────────────
        elif sig == -1:
            if active_leg == 1 and not np.isnan(ce_p):
                _close_position(ce_p, ts, "Close CE → Open PE")
            if active_leg == 0 and not np.isnan(pe_p):
                _open_position(-1, pe_p, ts)

        # ── Mark-to-market equity ─────────────────────────────────────────────
        if active_leg == 1 and not np.isnan(ce_p) and ce_p > 0:
            mtm = cash + (ce_p - entry_price) * n_lots * lot_size
        elif active_leg == -1 and not np.isnan(pe_p) and pe_p > 0:
            mtm = cash + (pe_p - entry_price) * n_lots * lot_size
        else:
            mtm = cash
        equities.append(mtm)

    # ── Force-close at last bar ────────────────────────────────────────────────
    if active_leg != 0:
        last_p = (float(ce_price_s.iloc[-1]) if active_leg == 1 else float(pe_price_s.iloc[-1]))
        if not np.isnan(last_p):
            _close_position(last_p, df["timestamp"].iloc[-1], "End of Data")
            if equities:
                equities[-1] = cash

    while len(equities) < len(df):
        equities.append(equities[-1] if equities else capital)

    results = df[["timestamp"]].copy()
    results["equity"]       = equities[:len(df)]
    results["peak"]         = results["equity"].cummax()
    results["drawdown_pct"] = (results["equity"] - results["peak"]) / results["peak"] * 100
    return results, trades


def metrics(results: pd.DataFrame, trades: list, capital: float) -> dict:
    import math

    final = results["equity"].iloc[-1]
    pnl   = final - capital
    ret   = round(pnl / capital * 100, 2)

    # CAGR — clamp to ±10000% to avoid astronomical display
    n_days = max((results["timestamp"].iloc[-1] - results["timestamp"].iloc[0]).days, 1)
    ratio  = max(final / capital, 1e-10) if capital > 0 else 1.0
    try:
        cagr_raw = (ratio ** (365.0 / n_days) - 1) * 100
        cagr = round(max(min(cagr_raw, 10_000.0), -100.0), 2)
    except (OverflowError, ZeroDivisionError):
        cagr = 10_000.0 if ret > 0 else -100.0

    max_dd = round(results["drawdown_pct"].min(), 2)

    # Sharpe/Sortino on DAILY returns (not bar-by-bar — avoids intraday inflation)
    try:
        daily  = results.set_index("timestamp")["equity"].resample("D").last().dropna()
        d_rets = daily.pct_change().dropna()
        sharpe  = round(float(d_rets.mean() / d_rets.std() * math.sqrt(252)), 3) if len(d_rets) > 1 and d_rets.std() > 0 else 0.0
        neg     = d_rets[d_rets < 0]
        sortino = round(float(d_rets.mean() / neg.std() * math.sqrt(252)), 3) if len(neg) > 1 and neg.std() > 0 else 0.0
        sharpe  = max(min(sharpe,  999.0), -999.0)
        sortino = max(min(sortino, 999.0), -999.0)
    except Exception:
        sharpe = sortino = 0.0

    calmar = round(cagr / abs(max_dd), 2) if max_dd != 0 else 0.0
    calmar = max(min(calmar, 999.0), -999.0)

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp, gl = sum(wins), abs(sum(losses))

    return {
        "total_return":  ret,
        "total_pnl":     round(pnl, 2),
        "cagr":          cagr,
        "max_drawdown":  max_dd,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "calmar":        calmar,
        "total_trades":  len(trades),
        "win_rate":      round(len(wins)/len(trades)*100, 1) if trades else 0.0,
        "profit_factor": round(gp/gl, 2) if gl > 0 else (round(gp, 2) if gp > 0 else 0.0),
        "avg_win":       round(float(np.mean(wins)),   2) if wins   else 0.0,
        "avg_loss":      round(float(np.mean(losses)), 2) if losses else 0.0,
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

    # ── Trade Style ────────────────────────────────────────────────────────
    trade_style = st.radio(
        "Trade Style",
        ["📈 Intraday", "🌙 Carry Forward"],
        horizontal=True,
        help="Intraday: positions force-squared off at EOD each day.\n"
             "Carry Forward: positions held overnight until exit signal fires.",
    )
    is_intraday = trade_style == "📈 Intraday"

    if is_intraday:
        INTRADAY_TF = {
            "1 Min": "1", "3 Min": "3", "5 Min": "5",
            "10 Min": "10", "15 Min": "15", "25 Min": "25", "60 Min (1 Hr)": "60",
        }
        interval_lbl = st.selectbox("Timeframe", list(INTRADAY_TF.keys()), index=5,
                                     help="Intraday timeframes only")
        interval     = INTRADAY_TF[interval_lbl]
        c_eod1, c_eod2 = st.columns(2)
        with c_eod1:
            eod_hour   = st.number_input("EOD Exit Hour",   min_value=9,  max_value=15, value=15, step=1)
        with c_eod2:
            eod_minute = st.number_input("EOD Exit Minute", min_value=0,  max_value=59, value=15, step=5)
        import datetime as _dt2
        eod_exit_time = _dt2.time(int(eod_hour), int(eod_minute))
    else:
        CF_TF = {
            "5 Min": "5", "15 Min": "15", "25 Min": "25",
            "60 Min": "60", "Daily": "D",
        }
        interval_lbl = st.selectbox("Timeframe", list(CF_TF.keys()), index=4,
                                     help="Carry Forward — positions held overnight")
        interval     = CF_TF[interval_lbl]
        eod_exit_time = None

    bsp_tf = st.selectbox(
        "BSP Timeframe",
        ["Daily (Pine Default ✅)", "Same as Chart"],
        index=0,
        help=(
            "Daily (Pine Default): matches TradingView Pine Script exactly. "
            "Pine uses request.security(..., 'D', mf) — BSP is ALWAYS computed on Daily bars "
            "regardless of chart timeframe.\n\n"
            "Same as Chart: BSP computed on your selected candle interval."
        )
    )
    use_daily_bsp = "Daily" in bsp_tf

    c1, c2 = st.columns(2)
    with c1:
        default_from = (datetime.now() - timedelta(days=30)) if is_intraday else (datetime.now() - timedelta(days=365))
        from_date = st.date_input("From", default_from)
    with c2:
        to_date = st.date_input("To", datetime.now())

    if is_intraday and (to_date - from_date).days > 90:
        st.warning("⚠️ Intraday data: Dhan API allows max ~90 days per call. Consider shorter ranges for speed.")

    # ── Backtest Mode ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">🎯 Backtest Mode</div>', unsafe_allow_html=True)

    backtest_mode = st.radio(
        "Trade on",
        ["Index (Futures-style)", "Options (Real Data)",
         "Strike Chart (ATM CE+PE)", "CE/PE Alternating (Pine Script)"],
        index=0,
        help=(
            "Index mode: signals on Nifty index OHLCV.\n\n"
            "Options (Real Data): signals on index, trades real option premiums.\n\n"
            "Strike Chart: BSP computed on ATM CE and PE candles independently.\n\n"
            "CE/PE Alternating: mirrors Pine Script exactly — BUY → open CE; "
            "SELL → close CE + open PE; next BUY → close PE + open CE (infinite rotation)."
        )
    )
    trade_options      = backtest_mode == "Options (Real Data)"
    strike_chart_mode  = backtest_mode == "Strike Chart (ATM CE+PE)"
    alternating_mode   = backtest_mode == "CE/PE Alternating (Pine Script)"

    if strike_chart_mode:
        st.markdown('<div class="sidebar-header">📌 Strike Chart Settings</div>', unsafe_allow_html=True)
        sc_expiry_flag = st.radio("Expiry Type", ["WEEK", "MONTH"], horizontal=True,
                               help="Weekly or Monthly expiry contract", key="sc_expiry")
        sc_lots = st.number_input("Lots per leg", min_value=1, max_value=50, value=1,
                                  help="CE leg lots and PE leg lots are independent")
        st.info("BSP on ATM CE candles -> CE BUY trade. BSP on ATM PE candles -> PE BUY trade.")

    # Defaults for alternating mode variables (prevent NameError when mode not active)
    alt_expiry_flag   = "WEEK"
    alt_strike_offset = 0

    if alternating_mode:
        st.markdown('<div class="sidebar-header">🔁 CE/PE Alternating Settings</div>', unsafe_allow_html=True)
        alt_expiry_flag = st.radio("Expiry Type", ["WEEK", "MONTH"], horizontal=True,
                                   help="Weekly or Monthly expiry contract", key="alt_expiry")
        _offset_opts   = list(range(-5, 6))
        _offset_labels = ["ATM" if o == 0 else (f"ATM+{o}" if o > 0 else f"ATM{o}") for o in _offset_opts]
        alt_strike_lbl = st.select_slider(
            "Strike (CE & PE)", options=_offset_labels, value="ATM",
            help="Both CE and PE legs use the same offset from ATM"
        )
        alt_strike_offset = _offset_opts[_offset_labels.index(alt_strike_lbl)]
        st.info(
            "\U0001F4CC **State Machine (Pine Script exact):**\n\n"
            "\U0001F7E2 **BUY signal** \u2192 Open CE (buy call)\n\n"
            "\U0001F534 **SELL signal** \u2192 Close CE \u2192 Open PE (buy put)\n\n"
            "\U0001F7E2 **Next BUY** \u2192 Close PE \u2192 Open CE\n\n"
            "\u2026repeats indefinitely"
        )

    if trade_options:
        st.markdown('<div class="sidebar-header">📌 Option Legs (Spread / Hedge)</div>', unsafe_allow_html=True)

        expiry_flag = st.radio("Expiry Type", ["WEEK", "MONTH"], horizontal=True,
                               help="Weekly or Monthly expiry contract")

        n_legs = st.selectbox("Number of Legs", [1, 2, 3, 4], index=0,
                              help="1 = Single option | 2 = Spread | 3-4 = Complex (Iron Condor etc.)")

        # Preset templates
        PRESETS = {
            "Custom": None,
            "Bull Call Spread":  [("CALL","ATM",0,"BUY"),  ("CALL","ATM+1",1,"SELL")],
            "Bear Put Spread":   [("PUT", "ATM",0,"BUY"),  ("PUT", "ATM-1",-1,"SELL")],
            "Bull Put Spread":   [("PUT", "ATM-1",-1,"SELL"),("PUT","ATM-2",-2,"BUY")],
            "Bear Call Spread":  [("CALL","ATM+1",1,"SELL"),("CALL","ATM+2",2,"BUY")],
            "Long Strangle":     [("CALL","ATM+1",1,"BUY"), ("PUT","ATM-1",-1,"BUY")],
            "Short Strangle":    [("CALL","ATM+1",1,"SELL"),("PUT","ATM-1",-1,"SELL")],
            "Long Straddle":     [("CALL","ATM",0,"BUY"),  ("PUT","ATM",0,"BUY")],
            "Iron Condor":       [("PUT","ATM-2",-2,"BUY"),("PUT","ATM-1",-1,"SELL"),
                                  ("CALL","ATM+1",1,"SELL"),("CALL","ATM+2",2,"BUY")],
        }

        preset = st.selectbox("📋 Strategy Template", list(PRESETS.keys()), index=0)

        offsets_list  = list(range(-10, 11))
        offset_labels = [get_strike_label(o) for o in offsets_list]

        option_legs = []
        for i in range(n_legs):
            with st.expander(f"Leg {i+1}", expanded=True):
                # Auto-fill from preset
                if preset != "Custom" and PRESETS[preset] and i < len(PRESETS[preset]):
                    p_type, p_label, p_off, p_dir = PRESETS[preset][i]
                    def_type = 0 if p_type == "CALL" else 1
                    def_off  = p_label
                    def_dir  = 0 if p_dir == "BUY" else 1
                else:
                    def_type = 0; def_off = "ATM"; def_dir = 0

                c1, c2 = st.columns(2)
                with c1:
                    leg_type = st.radio(f"Type##leg{i}", ["CE (Call)", "PE (Put)"],
                                        index=def_type, horizontal=True, key=f"leg_type_{i}")
                with c2:
                    leg_dir  = st.radio(f"Direction##leg{i}", ["BUY", "SELL"],
                                        index=def_dir, horizontal=True, key=f"leg_dir_{i}")

                leg_strike_lbl = st.select_slider(
                    f"Strike##leg{i}",
                    options=offset_labels,
                    value=def_off if def_off in offset_labels else "ATM",
                    key=f"leg_strike_{i}"
                )
                leg_offset = offsets_list[offset_labels.index(leg_strike_lbl)]
                leg_lots   = st.number_input(f"Lots##leg{i}", min_value=1, max_value=50,
                                              value=1, key=f"leg_lots_{i}")

                option_legs.append({
                    "opt_type":    "CALL" if "CE" in leg_type else "PUT",
                    "direction":   leg_dir,          # "BUY" or "SELL"
                    "offset":      leg_offset,
                    "strike_lbl":  leg_strike_lbl,
                    "lots":        int(leg_lots),
                })

        # Backward compat single-leg variables
        opt_type      = option_legs[0]["opt_type"]
        strike_offset = option_legs[0]["offset"]

        # Strategy summary
        leg_summaries = " | ".join(
            f"{lg['direction']} {lg['lots']}L {lg['strike_lbl']} {lg['opt_type']}"
            for lg in option_legs
        )
        st.info(f"**Strategy:** {leg_summaries}")

    # ── Strategy Parameters ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">⚙️ Strategy Parameters</div>', unsafe_allow_html=True)

    pivot_length = st.slider("Pivot Lookback",  3, 20, 5)
    bsp_length   = st.slider("BSP Length",      5, 50, 21)
    bsp_buy_lvl  = st.number_input("BSP Buy Level",  value=0.08,  step=0.01, format="%.2f")
    bsp_sell_lvl = st.number_input("BSP Sell Level", value=-0.08, step=0.01, format="%.2f")
    vol_threshold = st.slider("Volume Threshold",   1.0, 3.0, 1.5, 0.1)
    vol_period    = st.slider("Volume SMA Period",   5,  30,  14)

    st.markdown('<div class="sidebar-header">🎚️ Signal Mode</div>', unsafe_allow_html=True)
    signal_mode = st.selectbox(
        "Signal Mode",
        ["Flip", "Level Hold", "BSP Only"],
        index=1,
        help=(
            "Flip: trade only on BSP level CROSS — fewest signals, can miss trades on long history\n"
            "Level Hold: enter while BSP above threshold, exit when BSP returns to 0 — balanced\n"
            "BSP Only: no EMA filter, pure BSP — most signals, good for long backtests"
        )
    )
    ema_filter = st.checkbox(
        "EMA Trend Filter",
        value=(signal_mode != "BSP Only"),
        help="Require EMA20 > EMA50 for longs (and vice-versa). Uncheck for more signals on longer histories.",
        disabled=(signal_mode == "BSP Only")
    )

    st.markdown('<div class="sidebar-header">📈 EMA Display</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        show_e20  = st.checkbox("EMA 20",  True)
        show_e50  = st.checkbox("EMA 50",  True)
    with c2:
        show_e100 = st.checkbox("EMA 100", True)
        show_e200 = st.checkbox("EMA 200", True)

    st.markdown('<div class="sidebar-header">💼 Capital & Sizing</div>', unsafe_allow_html=True)
    init_capital  = st.number_input("Initial Capital (₹)", value=500000, step=50000)
    sizing_mode   = st.radio("Position Sizing", ["% of Capital", "Fixed Lots"], horizontal=True)
    if sizing_mode == "% of Capital":
        pos_size_pct = st.slider("Position Size (%)", 5, 100, 50)
        fixed_lots   = None
    else:
        fixed_lots   = st.number_input("Fixed Lots per Trade", min_value=1, max_value=100, value=1, step=1)
        pos_size_pct = 100   # unused in fixed mode, placeholder
    comm_pct = st.number_input("Commission (%)", value=0.03, step=0.01, format="%.3f")

    st.markdown('<div class="sidebar-header">🔄 Trade Style</div>', unsafe_allow_html=True)
    trade_style = st.radio(
        "Position Carrying",
        ["📅 Intraday (MIS)", "📆 Carry Forward (NRML)"],
        index=0,
        help=(
            "Intraday (MIS): All open positions are force-closed at 15:15 IST each day.\n"
            "Carry Forward (NRML): Positions can be held overnight across multiple days."
        )
    )
    is_intraday = (trade_style == "📅 Intraday (MIS)")

    if is_intraday:
        eod_exit_time = st.time_input(
            "EOD Square-off Time",
            value=__import__("datetime").time(15, 15),
            help="Force-close all positions at this time each day (default 15:15 IST)"
        )
    else:
        eod_exit_time = None

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

    # ══════════════════════════════════════════════════════════════════════════
    # CE/PE ALTERNATING MODE — mirrors Pine Script state machine
    # ══════════════════════════════════════════════════════════════════════════
    if alternating_mode:
        fetch_interval = interval if interval != "D" else "25"
        fd = from_date.strftime("%Y-%m-%d")
        td = to_date.strftime("%Y-%m-%d")

        # ── 1. Fetch index OHLCV for signals ──────────────────────────────────
        with st.spinner(f"📡 Fetching {index_name} index data for signals…"):
            df = _fetcher.fetch_index_ohlcv(
                idx_cfg["security_id"],
                fd, td, interval,
                debug=debug_mode
            )
        if df is None or df.empty:
            st.error("❌ No index data. Check token / date range.")
            st.stop()

        # Compute indicators + signals on index
        df["ema20"]   = ema(df["close"], 20)
        df["ema50"]   = ema(df["close"], 50)
        df["ema100"]  = ema(df["close"], 100)
        df["ema200"]  = ema(df["close"], 200)
        df["vol_sma"] = df["volume"].rolling(vol_period).mean()

        if use_daily_bsp and interval != "D":
            with st.spinner("📅 Fetching daily bars for BSP…"):
                daily_df = _fetcher.fetch_index_ohlcv(
                    idx_cfg["security_id"], fd, td, interval="D", debug=False)
            df["bsp"] = calc_bsp_daily(df, daily_df, bsp_length)
        else:
            df["bsp"] = calc_bsp(df, bsp_length)

        df = generate_signals(df, bsp_buy_lvl, bsp_sell_lvl,
                              ema_filter=ema_filter, signal_mode=signal_mode)

        n_buy  = int((df["signal"] == 1).sum())
        n_sell = int((df["signal"] == -1).sum())
        st.success(f"✅ Index: **{len(df):,} bars** | Buy signals: **{n_buy}** | Sell signals: **{n_sell}**")

        # ── 2. Fetch ATM CE option data ────────────────────────────────────────
        with st.spinner(f"📡 Fetching {index_name} ATM{'+'+str(alt_strike_offset) if alt_strike_offset > 0 else ('-'+str(abs(alt_strike_offset)) if alt_strike_offset < 0 else '')} CE…"):
            ce_raw = _fetcher.fetch_rolling_option(
                idx_cfg["security_id"], alt_strike_offset, "CALL",
                fd, td, interval=fetch_interval,
                expiry_flag=alt_expiry_flag, debug=debug_mode
            )

        # ── 3. Fetch ATM PE option data ────────────────────────────────────────
        with st.spinner(f"📡 Fetching {index_name} ATM{'+'+str(alt_strike_offset) if alt_strike_offset > 0 else ('-'+str(abs(alt_strike_offset)) if alt_strike_offset < 0 else '')} PE…"):
            pe_raw = _fetcher.fetch_rolling_option(
                idx_cfg["security_id"], alt_strike_offset, "PUT",
                fd, td, interval=fetch_interval,
                expiry_flag=alt_expiry_flag, debug=debug_mode
            )

        # Fallback to Black-Scholes simulation if API returns nothing
        if ce_raw is None or ce_raw.empty:
            st.warning("⚠️ No CE data — using Black-Scholes simulation.")
            ce_raw = simulate_option_prices(df, alt_strike_offset, strike_gap, "CALL", 7, 0.15)
            ce_raw = ce_raw.rename(columns={"opt_price": "close"})[["timestamp", "close"]]
        if pe_raw is None or pe_raw.empty:
            st.warning("⚠️ No PE data — using Black-Scholes simulation.")
            pe_raw = simulate_option_prices(df, alt_strike_offset, strike_gap, "PUT", 7, 0.15)
            pe_raw = pe_raw.rename(columns={"opt_price": "close"})[["timestamp", "close"]]

        st.success(f"✅ CE: **{len(ce_raw):,} bars** | PE: **{len(pe_raw):,} bars**")

        # ── 4. Run alternating backtest ────────────────────────────────────────
        with st.spinner("📊 Running CE/PE alternating backtest…"):
            results, trades = run_backtest_alternating(
                df, ce_raw, pe_raw,
                capital=init_capital,
                size_pct=pos_size_pct / 100,
                comm_pct=comm_pct / 100,
                lot_size=lot_size,
                fixed_lots=fixed_lots,
                is_intraday=is_intraday,
                eod_exit_time=eod_exit_time,
            )
            m = metrics(results, trades, init_capital)

        # ── 5. Mode banner ────────────────────────────────────────────────────
        _strike_tag = ("ATM" if alt_strike_offset == 0 else
                       f"ATM+{alt_strike_offset}" if alt_strike_offset > 0 else f"ATM{alt_strike_offset}")
        st.info(
            f"🔁 **CE/PE Alternating Mode** · {index_name} · {_strike_tag} · "
            f"{interval_lbl} · {alt_expiry_flag} expiry · "
            f"{'MIS' if is_intraday else 'NRML'}"
        )

        # ── 6. KPI row ────────────────────────────────────────────────────────
        k = st.columns(7)
        k[0].metric("Net P&L",        f"₹{m['total_pnl']:,.0f}")
        k[1].metric("Total Return",   f"{m['total_return']:.1f}%")
        k[2].metric("CAGR",           f"{m['cagr']:.1f}%")
        k[3].metric("Win Rate",       f"{m['win_rate']:.1f}%")
        k[4].metric("Profit Factor",  f"{m['profit_factor']:.2f}")
        k[5].metric("Max Drawdown",   f"{m['max_drawdown']:.1f}%")
        k[6].metric("Sharpe",         f"{m['sharpe']:.2f}")

        # ── 7. CE / PE trade split ────────────────────────────────────────────
        ce_trades = [t for t in trades if t.get("leg") == "CE"]
        pe_trades = [t for t in trades if t.get("leg") == "PE"]
        col_ce, col_pe = st.columns(2)
        with col_ce:
            st.markdown("#### 📈 CE (Call) Trades")
            st.metric("Count", len(ce_trades))
            if ce_trades:
                ce_pnl  = sum(t["pnl"] for t in ce_trades)
                ce_wins = sum(1 for t in ce_trades if t["pnl"] > 0)
                st.metric("Total P&L", f"₹{ce_pnl:,.2f}")
                st.metric("Win Rate",  f"{ce_wins/len(ce_trades)*100:.1f}%")
        with col_pe:
            st.markdown("#### 📉 PE (Put) Trades")
            st.metric("Count", len(pe_trades))
            if pe_trades:
                pe_pnl  = sum(t["pnl"] for t in pe_trades)
                pe_wins = sum(1 for t in pe_trades if t["pnl"] > 0)
                st.metric("Total P&L", f"₹{pe_pnl:,.2f}")
                st.metric("Win Rate",  f"{pe_wins/len(pe_trades)*100:.1f}%")

        st.divider()

        # ── 8. Tabs ───────────────────────────────────────────────────────────
        atab1, atab2, atab3, atab4 = st.tabs(["📈 Chart", "📋 Trades", "📉 Equity & DD", "📑 Stats"])

        with atab1:
            # Main price chart with signals
            ph_a   = pivot_highs(df, pivot_length)
            pl_a   = pivot_lows(df, pivot_length)
            obs_a  = order_blocks(df, ph_a, pl_a, vol_threshold)
            fvgs_a = fair_value_gaps(df)

            fig_a = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.55, 0.20, 0.25], vertical_spacing=0.025,
                subplot_titles=("Price + SMC Levels", "Volume", "BSP Oscillator")
            )
            fig_a.add_trace(go.Candlestick(
                x=df["timestamp"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name=index_name,
                increasing_line_color="#00d4aa", decreasing_line_color="#ff4b6e"
            ), row=1, col=1)

            for show, col, name, color in [
                (show_e20,  "ema20",  "EMA 20",  "#ff4b6e"),
                (show_e50,  "ema50",  "EMA 50",  "#ff9900"),
                (show_e100, "ema100", "EMA 100", "#00bcd4"),
                (show_e200, "ema200", "EMA 200", "#3f8ef5"),
            ]:
                if show:
                    fig_a.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=name,
                                               line=dict(color=color, width=1)), row=1, col=1)

            for ob in obs_a[:60]:
                fc = "rgba(255,75,110,0.10)" if ob["type"]=="bearish" else "rgba(0,212,170,0.10)"
                bc = "#ff4b6e" if ob["type"]=="bearish" else "#00d4aa"
                x0 = df["timestamp"].iloc[ob["start"]]
                x1 = df["timestamp"].iloc[min(ob["end"], len(df)-1)]
                fig_a.add_hrect(y0=ob["btm"], y1=ob["top"], x0=x0, x1=x1,
                                fillcolor=fc, line_color=bc, line_width=0.5,
                                annotation_text="OB★" if ob["strong"] else "OB",
                                annotation_font_color=bc, annotation_font_size=8,
                                row=1, col=1)

            for fvg in fvgs_a[:40]:
                x0 = df["timestamp"].iloc[fvg["start"]]
                x1 = df["timestamp"].iloc[min(fvg["start"]+15, len(df)-1)]
                fig_a.add_hrect(y0=fvg["btm"], y1=fvg["top"], x0=x0, x1=x1,
                                fillcolor="rgba(150,0,255,0.09)", line_color="#9000ff", line_width=0.5,
                                annotation_text="FVG", annotation_font_color="#b060ff",
                                annotation_font_size=8, row=1, col=1)

            # Signal markers — colour by CE/PE direction
            buys_a  = df[df["signal"] ==  1]
            sells_a = df[df["signal"] == -1]
            fig_a.add_trace(go.Scatter(
                x=buys_a["timestamp"], y=buys_a["low"] * 0.998,
                mode="markers", name="BUY → CE",
                marker=dict(symbol="triangle-up", size=10, color="#00d4aa",
                            line=dict(color="#fff", width=1))
            ), row=1, col=1)
            fig_a.add_trace(go.Scatter(
                x=sells_a["timestamp"], y=sells_a["high"] * 1.002,
                mode="markers", name="SELL → PE",
                marker=dict(symbol="triangle-down", size=10, color="#ff4b6e",
                            line=dict(color="#fff", width=1))
            ), row=1, col=1)

            # Volume
            vcol = ["#00d4aa" if c >= o else "#ff4b6e" for c, o in zip(df["close"], df["open"])]
            fig_a.add_trace(go.Bar(x=df["timestamp"], y=df["volume"],
                                   name="Volume", marker_color=vcol, opacity=0.6), row=2, col=1)
            fig_a.add_trace(go.Scatter(x=df["timestamp"], y=df["vol_sma"],
                                       name="Vol SMA", line=dict(color="#ff9900", width=1)), row=2, col=1)

            # BSP
            bsp_col_a = ["#00d4aa" if v > 0 else "#ff4b6e" for v in df["bsp"].fillna(0)]
            fig_a.add_trace(go.Bar(x=df["timestamp"], y=df["bsp"],
                                   name="BSP", marker_color=bsp_col_a, opacity=0.85), row=3, col=1)
            fig_a.add_hline(y=bsp_buy_lvl,  line_color="#00d4aa", line_dash="dash", line_width=1, row=3, col=1)
            fig_a.add_hline(y=bsp_sell_lvl, line_color="#ff4b6e", line_dash="dash", line_width=1, row=3, col=1)
            fig_a.add_hline(y=0,            line_color="#444",    line_width=1,      row=3, col=1)

            fig_a.update_layout(
                height=850, template="plotly_dark", showlegend=True,
                xaxis_rangeslider_visible=False,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(t=35, b=20)
            )
            fig_a.update_xaxes(showgrid=True, gridcolor="#1e2130")
            fig_a.update_yaxes(showgrid=True, gridcolor="#1e2130")
            st.plotly_chart(fig_a, use_container_width=True)

        with atab2:
            st.markdown("#### Trade Log — CE/PE Alternating")
            if trades:
                tdf_a = pd.DataFrame(trades)
                tdf_a["cum_pnl"] = tdf_a["pnl"].cumsum().round(2)
                for col in ["entry_price","exit_price","pnl","return_pct","cum_pnl"]:
                    if col in tdf_a.columns:
                        tdf_a[col] = tdf_a[col].round(2)

                rename_a = {
                    "leg": "Leg", "entry_time": "Entry", "exit_time": "Exit",
                    "entry_price": "Entry ₹", "exit_price": "Exit ₹",
                    "qty": "Qty", "lots": "Lots", "pnl": "P&L ₹",
                    "cum_pnl": "Cum. P&L ₹", "return_pct": "Return %",
                    "exit_reason": "Reason"
                }
                display_a = tdf_a[[c for c in rename_a if c in tdf_a.columns]].rename(columns=rename_a)
                float_cols_a = display_a.select_dtypes(include="number").columns.tolist()
                fmt_a = {c: "{:.2f}" for c in float_cols_a}
                color_cols_a = [c for c in ["P&L ₹", "Cum. P&L ₹", "Return %"] if c in display_a.columns]

                def _leg_bg(row):
                    styles = [""] * len(row)
                    if "Leg" in row.index:
                        leg_idx = list(row.index).index("Leg")
                        if row["Leg"] == "CE":
                            styles[leg_idx] = "background-color:rgba(0,212,170,0.15)"
                        elif row["Leg"] == "PE":
                            styles[leg_idx] = "background-color:rgba(255,75,110,0.15)"
                    return styles

                styled_a = (display_a.style
                    .format(fmt_a)
                    .apply(_leg_bg, axis=1)
                    .applymap(
                        lambda v: ("color:#00d4aa" if isinstance(v,(int,float)) and v > 0
                                   else "color:#ff4b6e" if isinstance(v,(int,float)) and v < 0 else ""),
                        subset=color_cols_a
                    ))
                st.dataframe(styled_a, use_container_width=True, height=500)

                c1_a, c2_a, c3_a, c4_a = st.columns(4)
                c1_a.metric("Net P&L",    f"₹{tdf_a['pnl'].sum():,.2f}")
                c2_a.metric("Trades",     len(tdf_a))
                c3_a.metric("Avg/Trade",  f"₹{tdf_a['pnl'].mean():,.2f}")
                c4_a.metric("Best Trade", f"₹{tdf_a['pnl'].max():,.2f}")

                csv_a = tdf_a.to_csv(index=False).encode()
                st.download_button("⬇️ Download CSV", csv_a,
                                   f"{index_name}_alternating_trades.csv", "text/csv")
            else:
                st.info("No trades generated. Try loosening BSP levels or switching Signal Mode.")

        with atab3:
            fig_eq_a = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     row_heights=[0.65, 0.35], vertical_spacing=0.05,
                                     subplot_titles=("Equity Curve (₹)", "Drawdown (%)"))
            fig_eq_a.add_trace(go.Scatter(
                x=results["timestamp"], y=results["equity"],
                name="Portfolio", fill="tozeroy",
                line=dict(color="#00d4aa", width=2),
                fillcolor="rgba(0,212,170,0.08)"
            ), row=1, col=1)
            fig_eq_a.add_hline(y=init_capital, line_color="#555", line_dash="dash", row=1, col=1)
            fig_eq_a.add_trace(go.Scatter(
                x=results["timestamp"], y=results["drawdown_pct"],
                name="Drawdown", fill="tozeroy",
                line=dict(color="#ff4b6e", width=1),
                fillcolor="rgba(255,75,110,0.12)"
            ), row=2, col=1)

            # Shade CE trades green, PE trades red on equity curve
            if trades:
                for t in trades:
                    if pd.notna(t.get("entry_time")) and pd.notna(t.get("exit_time")):
                        fc = "rgba(0,212,170,0.05)" if t.get("leg") == "CE" else "rgba(255,75,110,0.05)"
                        fig_eq_a.add_vrect(x0=t["entry_time"], x1=t["exit_time"],
                                           fillcolor=fc, line_width=0, row=1, col=1)

            fig_eq_a.update_layout(
                height=500, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(t=35, b=20)
            )
            fig_eq_a.update_xaxes(showgrid=True, gridcolor="#1e2130")
            fig_eq_a.update_yaxes(showgrid=True, gridcolor="#1e2130")
            st.plotly_chart(fig_eq_a, use_container_width=True)

        with atab4:
            c1_s, c2_s = st.columns(2)
            with c1_s:
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
            with c2_s:
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
                r2_a = results.copy()
                r2_a["month"]  = r2_a["timestamp"].dt.to_period("M")
                monthly_a = r2_a.groupby("month")["equity"].last().pct_change() * 100
                monthly_a.index = monthly_a.index.astype(str)
                monthly_a = monthly_a.dropna()
                fig_m_a = go.Figure(go.Bar(
                    x=monthly_a.index, y=monthly_a.values,
                    marker_color=["#00d4aa" if v >= 0 else "#ff4b6e" for v in monthly_a.values],
                    text=[f"{v:.1f}%" for v in monthly_a.values], textposition="outside",
                    textfont=dict(size=9)
                ))
                fig_m_a.update_layout(
                    height=280, template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    xaxis=dict(tickangle=-45), margin=dict(t=10, b=50),
                    yaxis_title="Return (%)"
                )
                st.plotly_chart(fig_m_a, use_container_width=True)

        st.stop()  # Don't run standard pipeline

    # ══════════════════════════════════════════════════════════════════════════
    # STRIKE CHART MODE — BSP on ATM CE and PE candles independently
    # ══════════════════════════════════════════════════════════════════════════
    if strike_chart_mode:
        fetch_interval = interval if interval != "D" else "25"
        fd = from_date.strftime("%Y-%m-%d")
        td = to_date.strftime("%Y-%m-%d")

        with st.spinner("📡 Fetching ATM CE candle data (OI+Volume)…"):
            ce_raw = _fetcher.fetch_rolling_option(
                idx_cfg["security_id"], 0, "CALL",
                fd, td, interval=fetch_interval,
                expiry_flag=sc_expiry_flag, debug=debug_mode
            )
        with st.spinner("📡 Fetching ATM PE candle data (OI+Volume)…"):
            pe_raw = _fetcher.fetch_rolling_option(
                idx_cfg["security_id"], 0, "PUT",
                fd, td, interval=fetch_interval,
                expiry_flag=sc_expiry_flag, debug=debug_mode
            )

        if ce_raw is None or ce_raw.empty:
            st.error("❌ No CE data returned. Check token / date range.")
            st.stop()
        if pe_raw is None or pe_raw.empty:
            st.error("❌ No PE data returned. Check token / date range.")
            st.stop()

        st.success(f"✅ CE: **{len(ce_raw):,} bars** | PE: **{len(pe_raw):,} bars**")

        # Signal diagnostics preview
        _ce_bsp = calc_bsp_oiv(ce_raw, bsp_length)
        _pe_bsp = calc_bsp_oiv(pe_raw, bsp_length)
        with st.expander("🔍 Strike BSP Diagnostics", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CE BSP Min", round(float(_ce_bsp.min()), 4))
            c2.metric("CE BSP Max", round(float(_ce_bsp.max()), 4))
            c3.metric("PE BSP Min", round(float(_pe_bsp.min()), 4))
            c4.metric("PE BSP Max", round(float(_pe_bsp.max()), 4))

        with st.spinner("📊 Running dual-strike backtest…"):
            results, trades = run_backtest_dual(
                ce_raw, pe_raw,
                capital=init_capital,
                size_pct=pos_size_pct / 100,
                comm_pct=comm_pct / 100,
                lot_size=lot_size,
                buy_lvl=bsp_buy_lvl, sell_lvl=bsp_sell_lvl,
                bsp_length=bsp_length,
                signal_mode=signal_mode,
                ema_filter=ema_filter,
                fixed_lots=int(sc_lots),
                is_intraday=is_intraday,
                eod_exit_time=eod_exit_time
            )
            m = metrics(results, trades, init_capital)

        st.info(f"📌 **Strike Chart Mode** · {index_name} · ATM CE + PE | {interval_lbl} | {sc_expiry_flag}")

        # ── Metrics Banner ────────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Net P&L",        f"₹{m['total_pnl']:,.0f}")
        c2.metric("Total Return",   f"{m['total_return']*100:.1f}%")
        c3.metric("Total Trades",   m["total_trades"])
        c4.metric("Win Rate",       f"{m['win_rate']:.1f}%")
        c5.metric("Max Drawdown",   f"{m['max_drawdown']:.1f}%")
        c6.metric("Sharpe Ratio",   f"{m['sharpe']:.2f}")

        # ── CE / PE split ─────────────────────────────────────────────────
        ce_trades = [t for t in trades if t.get("leg") == "CE"]
        pe_trades = [t for t in trades if t.get("leg") == "PE"]
        col_ce, col_pe = st.columns(2)
        with col_ce:
            st.markdown("#### 📈 CE Trades")
            st.metric("Count", len(ce_trades))
            if ce_trades:
                ce_pnl = sum(t["pnl"] for t in ce_trades)
                ce_wins = sum(1 for t in ce_trades if t["pnl"] > 0)
                st.metric("P&L", f"₹{ce_pnl:,.0f}")
                st.metric("Win Rate", f"{ce_wins/len(ce_trades)*100:.1f}%")
        with col_pe:
            st.markdown("#### 📉 PE Trades")
            st.metric("Count", len(pe_trades))
            if pe_trades:
                pe_pnl = sum(t["pnl"] for t in pe_trades)
                pe_wins = sum(1 for t in pe_trades if t["pnl"] > 0)
                st.metric("P&L", f"₹{pe_pnl:,.0f}")
                st.metric("Win Rate", f"{pe_wins/len(pe_trades)*100:.1f}%")

        # ── Equity Curve ──────────────────────────────────────────────────
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=results["timestamp"], y=results["equity"],
            mode="lines", name="Equity", line=dict(color="#00d4b4", width=2)
        ))
        fig_eq.update_layout(
            title="Equity Curve — Strike Chart Mode",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            height=350, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Trade Log ────────────────────────────────────────────────────
        if trades:
            tdf = pd.DataFrame(trades)
            tdf["pnl"] = tdf["pnl"].round(2)
            st.dataframe(tdf, use_container_width=True, height=300)
            csv = tdf.to_csv(index=False).encode()
            st.download_button("⬇️ Download Trade Log", csv, "strike_chart_trades.csv", "text/csv")
        else:
            st.warning("⚠️ No trades generated. Try loosening BSP levels or switching Signal Mode.")

        st.stop()  # Don't run rest of standard pipeline

    # ══════════════════════════════════════════════════════════════════════════
    # STANDARD MODE (Index / Options)
    # ══════════════════════════════════════════════════════════════════════════

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

        # BSP Timeframe — Pine uses request.security(..., 'D', mf) by default
        if use_daily_bsp and interval != "D":
            with st.spinner("📅 Fetching Daily bars for BSP calculation (Pine-accurate)…"):
                daily_df = _fetcher.fetch_index_ohlcv(
                    idx_cfg["security_id"],
                    from_date.strftime("%Y-%m-%d"),
                    to_date.strftime("%Y-%m-%d"),
                    interval="D",
                    debug=False
                )
            df["bsp"] = calc_bsp_daily(df, daily_df, bsp_length)
            st.caption("📅 BSP computed on **Daily** timeframe (matches TradingView Pine Script)")
        else:
            df["bsp"] = calc_bsp(df, bsp_length)

        ph   = pivot_highs(df, pivot_length)
        pl   = pivot_lows(df, pivot_length)
        obs  = order_blocks(df, ph, pl, vol_threshold)
        fvgs = fair_value_gaps(df)
        df   = generate_signals(df, bsp_buy_lvl, bsp_sell_lvl, ema_filter=ema_filter, signal_mode=signal_mode)

    # ── 2b. Signal Diagnostics ───────────────────────────────────────────────
    total_bars   = len(df)
    n_buy_sigs   = int((df["signal"] == 1).sum())
    n_sell_sigs  = int((df["signal"] == -1).sum())
    bsp_min      = round(float(df["bsp"].min()), 4)
    bsp_max      = round(float(df["bsp"].max()), 4)

    with st.expander("🔍 Signal Diagnostics", expanded=(n_buy_sigs == 0 and n_sell_sigs == 0)):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Bars",   f"{total_bars:,}")
        c2.metric("Buy Signals",  n_buy_sigs,  delta=None if n_buy_sigs  > 0 else "⚠️ None")
        c3.metric("Sell Signals", n_sell_sigs, delta=None if n_sell_sigs > 0 else "⚠️ None")
        c4.metric("BSP Min",  bsp_min)
        c5.metric("BSP Max",  bsp_max)

        if n_buy_sigs == 0 and n_sell_sigs == 0:
            st.warning(
                f"**No signals generated.**\n\n"
                f"BSP range on this data: **{bsp_min:.4f} → {bsp_max:.4f}**\n\n"
                f"Your thresholds: Buy > **{bsp_buy_lvl}**, Sell < **{bsp_sell_lvl}**\n\n"
                f"**Fix options (try in order):**\n"
                f"1. Switch Signal Mode to **BSP Only** or **Level Hold** (sidebar)\n"
                f"2. Loosen BSP levels — set Buy to **{round(bsp_max*0.6,3)}** and Sell to **{round(bsp_min*0.6,3)}**\n"
                f"3. Uncheck **EMA Trend Filter**\n"
                f"4. Increase BSP Length (longer smoothing)"
            )
            # Auto-suggest levels based on actual BSP range
            suggested_buy  = round(bsp_max * 0.5, 3)
            suggested_sell = round(bsp_min * 0.5, 3)
            st.info(f"💡 Auto-suggested levels for this dataset: Buy **{suggested_buy}** / Sell **{suggested_sell}**")

    # ── 3. Fetch Option Legs ─────────────────────────────────────────────────
    leg_dfs = []   # one price-series df per leg, aligned to main df index
    if trade_options:
        fetch_interval = interval if interval != "D" else "25"
        fd = from_date.strftime("%Y-%m-%d")
        td = to_date.strftime("%Y-%m-%d")

        for i, leg in enumerate(option_legs):
            lbl = f"Leg {i+1}: {leg['direction']} {leg['lots']}L {leg['strike_lbl']} {leg['opt_type']}"
            with st.spinner(f"📡 Fetching {lbl}…"):
                raw = _fetcher.fetch_rolling_option(
                    idx_cfg["security_id"],
                    leg["offset"], leg["opt_type"],
                    fd, td,
                    interval=fetch_interval,
                    expiry_flag=expiry_flag,
                    debug=debug_mode
                )

            if raw is not None and not raw.empty:
                st.success(f"✅ {lbl} — {len(raw):,} bars")
                # Merge onto main df timestamps
                leg_price = raw[["timestamp","close"]].rename(
                    columns={"close": f"leg{i}_price"}
                )
                merged = pd.merge_asof(
                    df[["timestamp"]].sort_values("timestamp"),
                    leg_price.sort_values("timestamp"),
                    on="timestamp", direction="nearest",
                    tolerance=pd.Timedelta("30min")
                )
                leg_dfs.append({**leg, "prices": merged.set_index("timestamp")[f"leg{i}_price"]})
            else:
                st.warning(f"⚠️ {lbl} — no data, falling back to Black-Scholes simulation.")
                # BS fallback for this leg
                sim = simulate_option_prices(
                    df, leg["offset"], strike_gap, leg["opt_type"], 7, 0.15
                )
                leg_dfs.append({**leg, "prices": sim.set_index("timestamp")["opt_price"]})

        # Build combined spread price column on df:
        # net_price = sum(signed_price per leg)
        # BUY leg  → pay premium  → negative cash at entry, positive at exit
        # SELL leg → receive prem → positive cash at entry, negative at exit
        df = df.set_index("timestamp")
        df["spread_price"] = 0.0
        for ld in leg_dfs:
            sign = 1 if ld["direction"] == "BUY" else -1
            p    = ld["prices"].reindex(df.index).ffill().fillna(0)
            df["spread_price"] += sign * p * ld["lots"]
        df = df.reset_index()

        # Legacy single-leg compat (used by chart subplot)
        if leg_dfs:
            df["opt_price"] = leg_dfs[0]["prices"].reindex(
                df.set_index("timestamp").index
            ).ffill().fillna(0).values

    # ── 4. Run Backtest ───────────────────────────────────────────────────────
    with st.spinner("📊 Running backtest…"):
        results, trades = run_backtest(
            df, init_capital, pos_size_pct / 100,
            comm_pct / 100, lot_size, trade_options,
            fixed_lots=fixed_lots,
            spread_legs=leg_dfs if trade_options else None,
            is_intraday=is_intraday,
            eod_exit_time=eod_exit_time
        )
        m = metrics(results, trades, init_capital)

    # ── 5. Mode Banner ────────────────────────────────────────────────────────
    style_tag = "📅 MIS (Intraday)" if is_intraday else "📆 NRML (Carry Forward)"
    eod_tag   = f" · EOD {eod_exit_time.strftime('%H:%M')}" if is_intraday and eod_exit_time else ""
    if trade_options:
        leg_summaries = " | ".join(
            f"{lg['direction']} {lg['lots']}L {lg['strike_lbl']} {lg['opt_type']}"
            for lg in option_legs
        )
        st.info(f"📌 **Options Mode** · {style_tag}{eod_tag} | {index_name} | {interval_lbl} | {leg_summaries} | {expiry_flag}")
    else:
        st.info(f"**{style_tag}**{eod_tag} · Index Mode | {index_name} | {interval_lbl} | Lot Size: {lot_size}")

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
            tdf = pd.DataFrame(trades)

            # Round core numeric columns to 2dp
            for col in ["entry_price","exit_price","pnl","return_pct"]:
                if col in tdf.columns:
                    tdf[col] = tdf[col].round(2)

            # Cumulative P&L
            tdf["cum_pnl"] = tdf["pnl"].cumsum().round(2)

            # Detect leg breakdown columns (leg1_ATM_CALL etc.)
            leg_cols = [c for c in tdf.columns if c.startswith("leg") and "_" in c[3:]]

            # Core columns always shown
            base_cols = ["entry_time","exit_time","entry_price","exit_price",
                         "qty","lots","pnl","cum_pnl","return_pct","exit_reason"]
            # Insert leg breakdown after exit_price if present
            all_cols = ["entry_time","exit_time","entry_price","exit_price",
                        "qty","lots"] + leg_cols + ["pnl","cum_pnl","return_pct","exit_reason"]
            all_cols = [c for c in all_cols if c in tdf.columns]

            rename_map = {
                "entry_time":  "Entry",   "exit_time":   "Exit",
                "entry_price": "Net Entry ₹" if len(leg_cols) > 0 else "Entry ₹",
                "exit_price":  "Net Exit ₹"  if len(leg_cols) > 0 else "Exit ₹",
                "qty":         "Qty",     "lots":        "Scale Lots" if len(leg_cols) > 0 else "Lots",
                "pnl":         "Net P&L ₹",  "cum_pnl":     "Cum. P&L ₹",
                "return_pct":  "Return %",   "exit_reason": "Reason",
            }
            # Pretty-name leg columns: leg1_ATM+1_CALL → "L1 ATM+1 CE ₹"
            for lc in leg_cols:
                parts = lc.split("_", 1)
                leg_num = parts[0].replace("leg", "L")
                rest    = parts[1].replace("_CALL", " CE").replace("_PUT", " PE")
                rename_map[lc] = f"{leg_num} {rest} ₹"

            display = tdf[all_cols].rename(columns=rename_map)

            # All float columns → 2dp format
            float_cols = display.select_dtypes(include="number").columns.tolist()
            fmt_dict   = {c: "{:.2f}" for c in float_cols}

            # Color: green for positive, red for negative
            color_cols = [rename_map.get("pnl","Net P&L ₹"),
                          rename_map.get("cum_pnl","Cum. P&L ₹"),
                          rename_map.get("return_pct","Return %")]
            color_cols += [rename_map[lc] for lc in leg_cols if lc in rename_map]
            color_cols  = [c for c in color_cols if c in display.columns]

            styled = display.style.format(fmt_dict).applymap(
                lambda v: ("color:#00d4aa" if isinstance(v,(int,float)) and v > 0
                           else "color:#ff4b6e" if isinstance(v,(int,float)) and v < 0
                           else ""),
                subset=color_cols
            )

            st.dataframe(styled, use_container_width=True, height=500)

            # Summary bar
            total_pnl = tdf["pnl"].sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net P&L", f"₹{total_pnl:,.2f}",
                      delta=f"{'▲' if total_pnl>0 else '▼'} {total_pnl/init_capital*100:.2f}%")
            c2.metric("Trades",        len(tdf))
            c3.metric("Avg P&L/Trade", f"₹{tdf['pnl'].mean():,.2f}")
            c4.metric("Best Trade",    f"₹{tdf['pnl'].max():,.2f}")

            # Leg-level P&L summary for spreads
            if leg_cols:
                st.markdown("##### 📊 Leg-wise P&L Summary")
                leg_summary_cols = st.columns(len(leg_cols))
                for j, lc in enumerate(leg_cols):
                    nice = rename_map.get(lc, lc)
                    total = tdf[lc].sum()
                    leg_summary_cols[j].metric(nice, f"₹{total:,.2f}",
                        delta=f"{'▲' if total>0 else '▼'}")

            csv = tdf.to_csv(index=False)
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
