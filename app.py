"""
NYZTrade SMC Liquidity Lens Backtester
All-in-one Streamlit app | Dhan API | Built for NIYAS
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
warnings.filterwarnings('ignore')

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
        letter-spacing: 0.1em; color: #666; margin-top: 1.1rem; margin-bottom: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] { background: #1a1d29; border-radius: 6px; border: 1px solid #2d3139; }
    .stTabs [aria-selected="true"] { background: #0052cc; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DHAN API
# ══════════════════════════════════════════════════════════════════════════════

EXCHANGE_MAP = {
    "NSE": "NSE_EQ", "BSE": "BSE_EQ",
    "NSE_FNO": "NSE_FNO", "BSE_FNO": "BSE_FNO",
    "MCX": "MCX_COMM", "NSE_CURRENCY": "NSE_CURRENCY",
}
INTRADAY_INTERVALS = {"1", "5", "15", "25", "60"}
DHAN_BASE = "https://api.dhan.co/v2"


def _dhan_headers(client_id: str, token: str) -> dict:
    return {
        "access-token": token,
        "client-id": client_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _parse_ohlcv(data: dict) -> pd.DataFrame | None:
    """Parse Dhan API OHLCV response dict → DataFrame."""
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


def fetch_dhan_data(
    client_id: str,
    access_token: str,
    security_id: str,
    exchange: str,
    instrument_type: str,
    from_date: str,
    to_date: str,
    interval: str,
) -> pd.DataFrame | None:
    """
    Fetch historical OHLCV from Dhan API.
    Auto-paginates intraday in 90-day chunks.
    """
    headers = _dhan_headers(client_id, access_token)
    segment = EXCHANGE_MAP.get(exchange, exchange)

    if interval in INTRADAY_INTERVALS:
        # ── Intraday: paginate in 90-day windows ──────────────────────────────
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt   = datetime.strptime(to_date,   "%Y-%m-%d")
        total_days = max((to_dt - from_dt).days, 1)
        fetched_days = 0
        all_dfs = []

        prog = st.progress(0, text="Fetching intraday data from Dhan…")
        chunk_start = from_dt

        while chunk_start <= to_dt:
            chunk_end = min(chunk_start + timedelta(days=90), to_dt)
            payload = {
                "securityId":       str(security_id),
                "exchangeSegment":  segment,
                "instrument":       instrument_type,
                "interval":         interval,
                "fromDate":         chunk_start.strftime("%Y-%m-%d"),
                "toDate":           chunk_end.strftime("%Y-%m-%d"),
            }
            try:
                r = requests.post(f"{DHAN_BASE}/charts/intraday",
                                  json=payload, headers=headers, timeout=30)
                r.raise_for_status()
                chunk_df = _parse_ohlcv(r.json())
                if chunk_df is not None:
                    all_dfs.append(chunk_df)
            except requests.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
                return None
            except Exception as e:
                st.error(f"Request failed: {e}")
                return None

            fetched_days += (chunk_end - chunk_start).days
            prog.progress(
                min(fetched_days / total_days, 1.0),
                text=f"Fetching {chunk_start.date()} → {chunk_end.date()}"
            )
            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(0.25)

        prog.empty()
        if not all_dfs:
            return None
        return (pd.concat(all_dfs)
                  .drop_duplicates("timestamp")
                  .sort_values("timestamp")
                  .reset_index(drop=True))

    else:
        # ── Daily / EOD ───────────────────────────────────────────────────────
        payload = {
            "securityId":       str(security_id),
            "exchangeSegment":  segment,
            "instrument":       instrument_type,
            "fromDate":         from_date,
            "toDate":           to_date,
            "expiryCode":       0,
        }
        try:
            r = requests.post(f"{DHAN_BASE}/charts/historical",
                              json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            return _parse_ohlcv(r.json())
        except requests.HTTPError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_bsp(df: pd.DataFrame, length: int) -> pd.Series:
    """BSP = sum(ad, n) / sum(vol, n)  where ad = ((2C-L-H)/(H-L))*V"""
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    ad = ((2 * df["close"] - df["low"] - df["high"]) / hl) * df["volume"]
    return (ad.rolling(length).sum() / df["volume"].rolling(length).sum()).fillna(0)


def pivot_highs(df: pd.DataFrame, length: int) -> list[dict]:
    pivots, n = [], len(df)
    for i in range(length, n - length):
        if df["high"].iloc[i] == df["high"].iloc[i - length: i + length + 1].max():
            pivots.append({"idx": i, "price": df["high"].iloc[i],
                           "open": df["open"].iloc[i], "close": df["close"].iloc[i],
                           "volume": df["volume"].iloc[i]})
    return pivots


def pivot_lows(df: pd.DataFrame, length: int) -> list[dict]:
    pivots, n = [], len(df)
    for i in range(length, n - length):
        if df["low"].iloc[i] == df["low"].iloc[i - length: i + length + 1].min():
            pivots.append({"idx": i, "price": df["low"].iloc[i],
                           "open": df["open"].iloc[i], "close": df["close"].iloc[i],
                           "volume": df["volume"].iloc[i]})
    return pivots


def order_blocks(df: pd.DataFrame, ph: list, pl: list,
                 vol_threshold: float = 1.5) -> list[dict]:
    """Bearish OBs at swing highs, bullish OBs at swing lows."""
    obs = []
    vol_sma = df["volume"].rolling(14).mean()

    for p in ph:
        i = p["idx"]
        strong = p["volume"] > vol_sma.iloc[i] * vol_threshold if pd.notna(vol_sma.iloc[i]) else False
        end = next((j for j in range(i + 1, len(df)) if df["close"].iloc[j] > p["price"]), len(df) - 1)
        obs.append({"type": "bearish", "top": p["price"],
                    "btm": max(p["open"], p["close"]),
                    "start": i, "end": end, "strong": strong})

    for p in pl:
        i = p["idx"]
        strong = p["volume"] > vol_sma.iloc[i] * vol_threshold if pd.notna(vol_sma.iloc[i]) else False
        end = next((j for j in range(i + 1, len(df)) if df["close"].iloc[j] < p["price"]), len(df) - 1)
        obs.append({"type": "bullish", "top": min(p["open"], p["close"]),
                    "btm": p["price"],
                    "start": i, "end": end, "strong": strong})
    return obs


def fair_value_gaps(df: pd.DataFrame, min_gap_pct: float = 0.001) -> list[dict]:
    """3-candle FVG pattern."""
    fvgs = []
    for i in range(2, len(df)):
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            mid = (df["low"].iloc[i] + df["high"].iloc[i - 2]) / 2
            if (df["low"].iloc[i] - df["high"].iloc[i - 2]) / mid > min_gap_pct:
                fvgs.append({"type": "bullish", "top": df["low"].iloc[i],
                              "btm": df["high"].iloc[i - 2], "start": i - 2})
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            mid = (df["low"].iloc[i - 2] + df["high"].iloc[i]) / 2
            if (df["low"].iloc[i - 2] - df["high"].iloc[i]) / mid > min_gap_pct:
                fvgs.append({"type": "bearish", "top": df["low"].iloc[i - 2],
                              "btm": df["high"].iloc[i], "start": i - 2})
    return fvgs


# ══════════════════════════════════════════════════════════════════════════════
# SIGNALS & BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(df: pd.DataFrame, buy_lvl: float, sell_lvl: float) -> pd.DataFrame:
    df = df.copy()
    long_c = (df["bsp"] > buy_lvl) & (df["close"] > df["ema20"]) & (df["ema20"] > df["ema50"])
    exit_c = (df["bsp"] < sell_lvl) & (df["close"] < df["ema20"]) & (df["ema20"] < df["ema50"])
    df["signal"] = np.where(long_c, 1, np.where(exit_c, -1, 0))
    return df


def run_backtest(df: pd.DataFrame, capital: float,
                 size_pct: float, comm_pct: float) -> tuple[pd.DataFrame, list]:
    cash, pos, entry_price, entry_time = capital, 0.0, 0.0, None
    in_trade, equities, trades = False, [], []

    for _, row in df.iterrows():
        sig, price, ts = row["signal"], row["close"], row["timestamp"]

        if sig == 1 and not in_trade:
            invest = cash * size_pct
            comm = invest * comm_pct
            pos = (invest - comm) / price
            cash -= invest
            entry_price, entry_time, in_trade = price, ts, True

        elif sig == -1 and in_trade:
            gross = pos * price
            comm = gross * comm_pct
            pnl = gross - comm - pos * entry_price
            cash += gross - comm
            trades.append({
                "entry_time": entry_time, "exit_time": ts,
                "entry_price": round(entry_price, 2), "exit_price": round(price, 2),
                "qty": round(pos, 4), "pnl": round(pnl, 2),
                "return_pct": round((price / entry_price - 1) * 100, 3),
                "exit_reason": "Signal",
            })
            pos, in_trade = 0.0, False

        equities.append(cash + pos * price)

    # Close open trade at last bar
    if in_trade and pos > 0:
        last_price = df["close"].iloc[-1]
        gross = pos * last_price
        comm = gross * comm_pct
        pnl = gross - comm - pos * entry_price
        cash += gross - comm
        trades.append({
            "entry_time": entry_time, "exit_time": df["timestamp"].iloc[-1],
            "entry_price": round(entry_price, 2), "exit_price": round(last_price, 2),
            "qty": round(pos, 4), "pnl": round(pnl, 2),
            "return_pct": round((last_price / entry_price - 1) * 100, 3),
            "exit_reason": "End of Data",
        })

    results = df[["timestamp"]].copy()
    results["equity"] = equities
    results["peak"] = results["equity"].cummax()
    results["drawdown_pct"] = (results["equity"] - results["peak"]) / results["peak"] * 100
    return results, trades


def performance_metrics(results: pd.DataFrame, trades: list, capital: float) -> dict:
    final = results["equity"].iloc[-1]
    total_pnl = final - capital
    total_ret = total_pnl / capital * 100
    n_days = (results["timestamp"].iloc[-1] - results["timestamp"].iloc[0]).days
    cagr = ((final / capital) ** (365 / max(n_days, 1)) - 1) * 100
    max_dd = results["drawdown_pct"].min()
    rets = results["equity"].pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    neg = rets[rets < 0]
    sortino = rets.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp, gl = sum(wins), abs(sum(losses))

    return {
        "total_return": total_ret, "total_pnl": total_pnl, "cagr": cagr,
        "max_drawdown": max_dd, "sharpe_ratio": sharpe, "sortino_ratio": sortino,
        "calmar_ratio": calmar, "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🔑 Dhan API Credentials")
    client_id    = st.text_input("Client ID", type="password")
    access_token = st.text_input("Access Token", type="password")

    st.markdown('<div class="sidebar-header">📈 Symbol Settings</div>', unsafe_allow_html=True)
    exchange      = st.selectbox("Exchange", ["NSE", "BSE", "NSE_FNO", "BSE_FNO", "MCX"], index=0)
    security_id   = st.text_input("Security ID (Dhan numeric ID)", value="1333",
                                   help="e.g. 1333 = HDFCBANK NSE | 13 = NIFTY")
    symbol_name   = st.text_input("Display Name", value="HDFCBANK")
    instrument    = st.selectbox("Instrument Type", ["EQUITY", "FUTIDX", "OPTIDX", "FUTCOM"], index=0)

    interval_map = {"1 Min":"1","5 Min":"5","15 Min":"15","25 Min":"25","60 Min":"60","Daily":"D"}
    interval_lbl = st.selectbox("Timeframe", list(interval_map.keys()), index=3)
    interval     = interval_map[interval_lbl]

    c1, c2 = st.columns(2)
    with c1: from_date = st.date_input("From", datetime.now() - timedelta(days=365))
    with c2: to_date   = st.date_input("To",   datetime.now())

    st.markdown('<div class="sidebar-header">⚙️ Strategy Parameters</div>', unsafe_allow_html=True)
    pivot_length  = st.slider("Pivot Lookback",            3, 20,  5)
    bsp_length    = st.slider("BSP Length",                5, 50, 21)
    bsp_buy_lvl   = st.number_input("BSP Buy Level",  value=0.08, step=0.01, format="%.2f")
    bsp_sell_lvl  = st.number_input("BSP Sell Level", value=-0.08, step=0.01, format="%.2f")
    vol_threshold = st.slider("Volume Threshold Multiplier", 1.0, 3.0, 1.5, 0.1)
    vol_period    = st.slider("Volume SMA Period",           5,  30,  14)

    st.markdown('<div class="sidebar-header">📊 EMA Overlay</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        show_e20  = st.checkbox("EMA 20",  True)
        show_e50  = st.checkbox("EMA 50",  True)
    with c2:
        show_e100 = st.checkbox("EMA 100", True)
        show_e200 = st.checkbox("EMA 200", True)

    st.markdown('<div class="sidebar-header">💼 Backtest Settings</div>', unsafe_allow_html=True)
    init_capital  = st.number_input("Initial Capital (₹)", value=100000, step=10000)
    pos_size_pct  = st.slider("Position Size (%)", 5, 100, 100)
    comm_pct      = st.number_input("Commission (%)", value=0.03, step=0.01, format="%.3f")

    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 📡 NYZTrade · SMC Liquidity Lens Backtest")
st.caption("Smart Money Concepts + BSP Oscillator + FVG Detection | Indian Markets | Dhan API")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    if not client_id or not access_token:
        st.error("⚠️ Enter Dhan API credentials in the sidebar.")
        st.stop()

    # ── 1. Fetch Data ─────────────────────────────────────────────────────────
    with st.spinner("📡 Fetching data from Dhan API…"):
        df = fetch_dhan_data(
            client_id, access_token, security_id, exchange, instrument,
            from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"), interval
        )

    if df is None or df.empty:
        st.error("❌ No data returned. Check credentials, Security ID, and date range.")
        st.stop()

    st.success(f"✅ Loaded **{len(df):,} bars** for **{symbol_name}** ({interval_lbl})")

    # ── 2. Indicators ─────────────────────────────────────────────────────────
    with st.spinner("⚙️ Computing indicators & running backtest…"):
        df["ema20"]   = ema(df["close"], 20)
        df["ema50"]   = ema(df["close"], 50)
        df["ema100"]  = ema(df["close"], 100)
        df["ema200"]  = ema(df["close"], 200)
        df["vol_sma"] = df["volume"].rolling(vol_period).mean()
        df["bsp"]     = calc_bsp(df, bsp_length)

        ph = pivot_highs(df, pivot_length)
        pl = pivot_lows(df, pivot_length)
        obs  = order_blocks(df, ph, pl, vol_threshold)
        fvgs = fair_value_gaps(df)

        df = generate_signals(df, bsp_buy_lvl, bsp_sell_lvl)
        results, trades = run_backtest(df, init_capital, pos_size_pct / 100, comm_pct / 100)
        m = performance_metrics(results, trades, init_capital)

    # ── 3. KPI Cards ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Performance Summary")
    k = st.columns(6)
    k[0].metric("Total Return",  f"{m['total_return']:.1f}%",    delta=f"₹{m['total_pnl']:,.0f}")
    k[1].metric("Profit Factor", f"{m['profit_factor']:.2f}")
    k[2].metric("Win Rate",      f"{m['win_rate']:.1f}%")
    k[3].metric("Max Drawdown",  f"{m['max_drawdown']:.1f}%",    delta_color="inverse")
    k[4].metric("Total Trades",  str(m['total_trades']))
    k[5].metric("Sharpe Ratio",  f"{m['sharpe_ratio']:.2f}")
    st.divider()

    # ── 4. Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "📋 Trades", "📉 Equity & Drawdown", "📑 Stats"])

    # ─ Tab 1: Chart ───────────────────────────────────────────────────────────
    with tab1:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.60, 0.20, 0.20],
            vertical_spacing=0.03,
            subplot_titles=("Price + SMC Levels", "Volume", "BSP Oscillator")
        )

        # Candles
        fig.add_trace(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
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
                                          line=dict(color=color, width=1), opacity=0.85), row=1, col=1)

        # Order Blocks (max 60)
        for ob in obs[:60]:
            fc = "rgba(255,75,110,0.12)" if ob["type"] == "bearish" else "rgba(0,212,170,0.12)"
            bc = "#ff4b6e"              if ob["type"] == "bearish" else "#00d4aa"
            x0 = df["timestamp"].iloc[ob["start"]]
            x1 = df["timestamp"].iloc[min(ob["end"], len(df) - 1)]
            fig.add_hrect(y0=ob["btm"], y1=ob["top"], x0=x0, x1=x1,
                          fillcolor=fc, line_color=bc, line_width=0.5,
                          annotation_text="OB★" if ob["strong"] else "OB",
                          annotation_font_color=bc, annotation_font_size=8,
                          row=1, col=1)

        # FVG Zones (max 40)
        for fvg in fvgs[:40]:
            x0 = df["timestamp"].iloc[fvg["start"]]
            x1 = df["timestamp"].iloc[min(fvg["start"] + 15, len(df) - 1)]
            fig.add_hrect(y0=fvg["btm"], y1=fvg["top"], x0=x0, x1=x1,
                          fillcolor="rgba(150,0,255,0.10)",
                          line_color="#9000ff", line_width=0.5,
                          annotation_text="FVG",
                          annotation_font_color="#b060ff", annotation_font_size=8,
                          row=1, col=1)

        # Buy / Sell markers
        buys  = df[df["signal"] ==  1]
        sells = df[df["signal"] == -1]
        fig.add_trace(go.Scatter(x=buys["timestamp"],  y=buys["low"]  * 0.998,
                                  mode="markers", name="BUY",
                                  marker=dict(symbol="triangle-up",   size=9, color="#00d4aa")), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["high"] * 1.002,
                                  mode="markers", name="EXIT",
                                  marker=dict(symbol="triangle-down", size=9, color="#ff4b6e")), row=1, col=1)

        # Volume bars
        vcol = ["#00d4aa" if c >= o else "#ff4b6e" for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], name="Volume",
                              marker_color=vcol, opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vol_sma"], name="Vol SMA",
                                  line=dict(color="#ff9900", width=1)), row=2, col=1)

        # BSP
        bsp_col = ["#00d4aa" if v > 0 else "#ff4b6e" for v in df["bsp"].fillna(0)]
        fig.add_trace(go.Bar(x=df["timestamp"], y=df["bsp"], name="BSP",
                              marker_color=bsp_col, opacity=0.85), row=3, col=1)
        fig.add_hline(y=bsp_buy_lvl,  line_color="#00d4aa", line_dash="dash", line_width=1, row=3, col=1)
        fig.add_hline(y=bsp_sell_lvl, line_color="#ff4b6e", line_dash="dash", line_width=1, row=3, col=1)
        fig.add_hline(y=0,            line_color="#444",    line_width=1,      row=3, col=1)

        fig.update_layout(
            height=820, template="plotly_dark", showlegend=True,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            margin=dict(t=35, b=20),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#1e2130")
        fig.update_yaxes(showgrid=True, gridcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)

    # ─ Tab 2: Trades ──────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Trade Log")
        if trades:
            tdf = pd.DataFrame(trades)
            tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
            tdf["exit_time"]  = pd.to_datetime(tdf["exit_time"])

            styled = (
                tdf[["entry_time","exit_time","entry_price","exit_price","qty","pnl","return_pct","exit_reason"]]
                .rename(columns={"entry_time":"Entry","exit_time":"Exit",
                                  "entry_price":"Entry ₹","exit_price":"Exit ₹",
                                  "qty":"Qty","pnl":"P&L ₹","return_pct":"Return %","exit_reason":"Reason"})
                .style.applymap(lambda v: "color:#00d4aa" if v > 0 else "color:#ff4b6e",
                                subset=["P&L ₹","Return %"])
            )
            st.dataframe(styled, use_container_width=True, height=500)

            csv = tdf.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv, f"{symbol_name}_trades.csv", "text/csv")
        else:
            st.info("No trades generated with current parameters.")

    # ─ Tab 3: Equity & Drawdown ───────────────────────────────────────────────
    with tab3:
        st.markdown("#### Equity Curve & Drawdown")
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

        fig2.update_layout(height=500, template="plotly_dark",
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            margin=dict(t=35, b=20))
        fig2.update_xaxes(showgrid=True, gridcolor="#1e2130")
        fig2.update_yaxes(showgrid=True, gridcolor="#1e2130")
        st.plotly_chart(fig2, use_container_width=True)

    # ─ Tab 4: Stats ───────────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Detailed Statistics")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Return Metrics**")
            for k, v in [
                ("Total Return (%)",  f"{m['total_return']:.2f}%"),
                ("Total P&L (₹)",     f"₹{m['total_pnl']:,.2f}"),
                ("CAGR (%)",          f"{m['cagr']:.2f}%"),
                ("Sharpe Ratio",      f"{m['sharpe_ratio']:.3f}"),
                ("Sortino Ratio",     f"{m['sortino_ratio']:.3f}"),
                ("Calmar Ratio",      f"{m['calmar_ratio']:.3f}"),
            ]:
                st.markdown(f"`{k}` &nbsp; **{v}**")
        with c2:
            st.markdown("**Trade Metrics**")
            for k, v in [
                ("Total Trades",     m['total_trades']),
                ("Win Rate (%)",     f"{m['win_rate']:.1f}%"),
                ("Profit Factor",    f"{m['profit_factor']:.3f}"),
                ("Avg Win (₹)",      f"₹{m['avg_win']:,.2f}"),
                ("Avg Loss (₹)",     f"₹{m['avg_loss']:,.2f}"),
                ("Max Drawdown (%)", f"{m['max_drawdown']:.2f}%"),
            ]:
                st.markdown(f"`{k}` &nbsp; **{v}**")

        # Monthly returns bar chart
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
                height=300, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                xaxis=dict(tickangle=-45), margin=dict(t=10, b=60),
                yaxis_title="Return (%)"
            )
            st.plotly_chart(fig3, use_container_width=True)

# ── Welcome screen ─────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; opacity:0.65;">
        <h2>📡 SMC Liquidity Lens Backtester</h2>
        <p style="font-size:1.05rem;">
            Enter your <b>Dhan API credentials</b> and strategy settings in the sidebar,
            then click <b>Run Backtest</b>.
        </p>
        <br>
        <table style="margin:auto; font-size:0.9rem; border-collapse:collapse; line-height:2;">
            <tr>
                <td style="padding:4px 20px;">✅ Pivot Order Blocks (Bearish & Bullish)</td>
                <td style="padding:4px 20px;">✅ BSP Money Flow Oscillator</td>
            </tr>
            <tr>
                <td style="padding:4px 20px;">✅ Fair Value Gap (FVG) Detection</td>
                <td style="padding:4px 20px;">✅ Volume-Weighted Signal Strength</td>
            </tr>
            <tr>
                <td style="padding:4px 20px;">✅ EMA 20 / 50 / 100 / 200 Filter</td>
                <td style="padding:4px 20px;">✅ Sharpe · Sortino · Calmar · CAGR</td>
            </tr>
            <tr>
                <td style="padding:4px 20px;">✅ Full Trade Log + CSV Export</td>
                <td style="padding:4px 20px;">✅ Monthly Returns Heatmap</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
