"""
Ravioli Stock Screener — finds the best ticker for Ravi to trade.
Scores stocks on trend strength, follow-through, and volatility quality.
Usage: python screener.py
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Watchlist — add/remove tickers as needed
# ---------------------------------------------------------------------------
WATCHLIST = [
    # Proven winners — backtested profitable at 1.5% risk (30 days, 2026-03-18)
    "SOUN",   # +$7,512  | Mid-cap AI/growth, high vol
    "LNAI",   # +$7,453  | Micro-cap, high volatility
    "TSLA",   # +$6,648  | Large-cap EV, best win rate (50%)
    "RBLX",   # +$5,564  | Gaming, low drawdown (2.71%)
    "SPOT",   # +$5,159  | Streaming, lowest drawdown (1.58%)
    "ROKU",   # +$3,001  | Streaming tech, moderate vol
    "META",   # +$1,875  | Large-cap tech, lowest drawdown (0.87%)
    "AVGO",   # +$1,688  | Semiconductor, stable
]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Average Directional Index — measures trend strength (0-100).
    Above 25 = trending, above 40 = strong trend."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, adjust=False).mean()
    return float(adx.iloc[-1]) if not adx.empty and pd.notna(adx.iloc[-1]) else 0.0


def compute_choppiness(df: pd.DataFrame, period: int = 14) -> float:
    """Choppiness Index (0-100). High = choppy/range-bound, Low = trending.
    Below 38 = strong trend, above 62 = very choppy."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(period).sum()
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    hl_range = highest - lowest
    hl_range = hl_range.replace(0, np.nan)

    chop = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
    return float(chop.iloc[-1]) if not chop.empty and pd.notna(chop.iloc[-1]) else 50.0


def compute_efficiency_ratio(close: pd.Series, period: int = 10) -> float:
    """Kaufman Efficiency Ratio. 1.0 = perfect trend, 0.0 = pure noise.
    Measures how much of the total bar-to-bar movement was directional."""
    if len(close) < period + 1:
        return 0.0
    direction = abs(close.iloc[-1] - close.iloc[-period])
    volatility = close.diff().abs().iloc[-period:].sum()
    if volatility == 0:
        return 0.0
    return float(direction / volatility)


def compute_follow_through(df: pd.DataFrame) -> float:
    """Measures how often a move in one bar continues in the next bar.
    High follow-through = trending behavior. Range: 0.0 - 1.0."""
    changes = df["Close"].diff()
    if len(changes) < 3:
        return 0.5
    same_dir = 0
    total = 0
    for i in range(2, len(changes)):
        if pd.notna(changes.iloc[i]) and pd.notna(changes.iloc[i - 1]):
            if changes.iloc[i] != 0 and changes.iloc[i - 1] != 0:
                total += 1
                if (changes.iloc[i] > 0) == (changes.iloc[i - 1] > 0):
                    same_dir += 1
    return same_dir / total if total > 0 else 0.5


def compute_volume_consistency(df: pd.DataFrame) -> float:
    """Ratio of mean to std of volume. High = consistent, Low = spiky.
    Consistent volume supports reliable signals."""
    vol = df["Volume"]
    if vol.std() == 0:
        return 1.0
    return float(vol.mean() / vol.std())


def compute_atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as percentage of price. Measures volatility magnitude."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    price = df["Close"].iloc[-1]
    if price == 0:
        return 0.0
    return float(atr.iloc[-1] / price * 100)


# ---------------------------------------------------------------------------
# Main screener
# ---------------------------------------------------------------------------
def compute_vwap_crosses(df: pd.DataFrame) -> float:
    """Average VWAP crosses per day. Fewer = more trending, more = choppy.
    This is the #1 predictor of Ravi profitability (r=-0.720)."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]
    typical = (high + low + close) / 3

    crosses_per_day = []
    for date, group in close.groupby(close.index.date):
        if len(group) < 50:
            continue
        day_typ = typical.loc[group.index]
        day_vol = vol.loc[group.index]
        cum_tp = (day_typ * day_vol).cumsum()
        cum_v = day_vol.cumsum().replace(0, np.nan)
        vwap = cum_tp / cum_v
        day_close = close.loc[group.index]
        above = day_close > vwap
        crosses = (above != above.shift(1)).sum()
        crosses_per_day.append(crosses)
    return float(np.mean(crosses_per_day)) if crosses_per_day else 30.0


def score_stock(symbol: str, df: pd.DataFrame) -> dict:
    """Score a single stock based on data-calibrated metrics.
    Weights derived from correlation analysis against 20-ticker backtest results."""
    if df.empty or len(df) < 100:
        return {"symbol": symbol, "score": 0, "reason": "Insufficient data"}

    # Use last 5 trading days of intraday data for current conditions
    last_5d = df.tail(5 * 390)  # ~5 days of 1-min bars

    chop = compute_choppiness(last_5d)
    efficiency = compute_efficiency_ratio(last_5d["Close"], period=60)
    follow = compute_follow_through(last_5d)
    vwap_crosses = compute_vwap_crosses(last_5d)
    atr_pct = compute_atr_percent(last_5d)
    adx = compute_adx(last_5d)

    # Score each component (0-100 scale) — calibrated against actual P&L

    # VWAP Crosses: strongest predictor (r=-0.720). Fewer = better.
    # Winners avg ~15, losers avg ~20+, AMC was 60
    if vwap_crosses <= 12:
        vwap_score = 100
    elif vwap_crosses <= 18:
        vwap_score = 100 - (vwap_crosses - 12) * 8
    elif vwap_crosses <= 25:
        vwap_score = 52 - (vwap_crosses - 18) * 5
    else:
        vwap_score = max(0, 20 - (vwap_crosses - 25) * 2)

    # Choppiness (r=-0.683): lower = more trending = better
    # Winners ~43-51, losers ~48-89
    chop_score = min(100, max(0, (55 - chop) * 5))


    # Efficiency (r=+0.667): higher = more directional = better
    # Winners ~0.13-0.16, losers ~0.04-0.14
    eff_score = min(100, max(0, efficiency * 500))

    # Follow-through (r=+0.520): higher = moves continue = better
    # Winners ~0.47-0.50, losers ~0.24-0.50
    follow_score = min(100, max(0, (follow - 0.40) * 500))

    # ADX: moderate is best. Sweet spot: 25-35.
    # Too high (>40) = violent whipsaw (SMCI/RIOT failed here), too low (<25) = no trend
    if adx < 20:
        adx_score = 0
    elif adx < 25:
        adx_score = (adx - 20) * 16  # ramp up to 80
    elif adx <= 35:
        adx_score = 80 + (adx - 25) * 2  # 80-100 sweet spot
    elif adx <= 40:
        adx_score = 100 - (adx - 35) * 15  # sharp drop: 100→25
    else:
        adx_score = max(0, 25 - (adx - 40) * 5)  # >40 tanks hard

    # Weighted final score — weights based on correlation + validation
    final_score = (
        vwap_score * 0.25 +      # #1 predictor (r=-0.720)
        chop_score * 0.25 +       # #2 predictor (r=-0.683)
        eff_score * 0.20 +        # #3 predictor (r=+0.667)
        follow_score * 0.10 +     # #4 predictor (r=+0.520)
        adx_score * 0.20          # Tightened — extreme ADX kills profitability
    )

    # Hard cap: extreme ADX = disqualified
    if adx > MAX_ADX:
        final_score = 0

    return {
        "symbol": symbol,
        "score": round(final_score, 1),
        "vwap_x": round(vwap_crosses, 1),
        "chop": round(chop, 1),
        "efficiency": round(efficiency, 3),
        "follow_through": round(follow, 3),
        "adx": round(adx, 1),
        "atr_pct": round(atr_pct, 2),
        "price": round(float(df["Close"].iloc[-1]), 2),
    }


def run_screener(watchlist: list[str] = None):
    if watchlist is None:
        watchlist = WATCHLIST

    print(f"\n{'='*70}")
    print(f"  RAVIOLI STOCK SCREENER")
    print(f"  Scanning {len(watchlist)} tickers for best trading conditions...")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}\n")

    results = []

    for sym in watchlist:
        try:
            print(f"  Scanning {sym}...", end=" ", flush=True)
            df = yf.download(sym, period="5d", interval="1m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                print("no data")
                continue
            result = score_stock(sym, df)
            results.append(result)
            print(f"score: {result['score']}")
        except Exception as e:
            print(f"error: {e}")

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Print results
    print(f"\n{'='*70}")
    print(f"  SCREENING RESULTS — RANKED BY RAVI COMPATIBILITY")
    print(f"{'='*70}\n")
    print(f"  {'Rank':<5} {'Ticker':<7} {'Score':<7} {'VWAP_X':<8} {'Chop':<7} {'Eff.':<7} {'Follow':<8} {'ADX':<7} {'ATR%':<7} {'Price':<10} {'Verdict'}")
    print(f"  {'-'*80}")

    for i, r in enumerate(results, 1):
        if "reason" in r:
            verdict = r["reason"]
        elif r["score"] >= 65:
            verdict = "STRONG BUY"
        elif r["score"] >= 50:
            verdict = "GOOD"
        elif r["score"] >= 35:
            verdict = "RISKY"
        else:
            verdict = "AVOID"

        symbol = r["symbol"]
        if r["score"] >= 65:
            marker = ">>>"
        elif r["score"] >= 50:
            marker = " > "
        else:
            marker = "   "

        print(
            f"  {marker}{i:<3} {symbol:<7} {r['score']:<7} "
            f"{r.get('vwap_x', 'N/A'):<8} {r.get('chop', 'N/A'):<7} "
            f"{r.get('efficiency', 'N/A'):<7} {r.get('follow_through', 'N/A'):<8} "
            f"{r.get('adx', 'N/A'):<7} {r.get('atr_pct', 'N/A'):<7} "
            f"${r.get('price', 'N/A'):<9} {verdict}"
        )

    print(f"\n  {'-'*80}")
    print(f"  LEGEND:")
    print(f"    VWAP_X: VWAP crosses/day (fewer = trending, #1 predictor)")
    print(f"    Chop: Choppiness (lower = more trending)")
    print(f"    Eff.: Directional efficiency (higher = cleaner moves)")
    print(f"    Follow: Move continuation rate (higher = trends sustain)")
    print(f"    ADX: Trend strength (25-35 sweet spot)")
    print(f"    ATR%: Volatility as % of price")
    print()

    if results and results[0]["score"] >= 50:
        top = results[0]
        print(f"  RECOMMENDATION: Run Ravi on {top['symbol']} (score: {top['score']})")
        print(f"  Current price: ${top.get('price', 'N/A')}")
    elif results:
        print(f"  WARNING: No strong candidates right now. Best option: {results[0]['symbol']} (score: {results[0]['score']})")
        print(f"  Consider waiting for better market conditions.")
    print()

    return results


# ---------------------------------------------------------------------------
# Auto-scan (called by bot_engine)
# ---------------------------------------------------------------------------
MIN_SCORE = 60
MAX_ADX = 50            # Hard cap — extreme ADX = violent whipsaw, auto-disqualify

def auto_scan(watchlist: list[str] = None) -> dict | None:
    """Scan all tickers and return the best pick above MIN_SCORE.
    Returns None if no ticker qualifies."""
    if watchlist is None:
        watchlist = WATCHLIST

    results = []
    for sym in watchlist:
        try:
            df = yf.download(sym, period="5d", interval="1m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                continue
            result = score_stock(sym, df)
            if result.get("score", 0) > 0:
                results.append(result)
        except Exception:
            continue

    if not results:
        return None

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    if best["score"] >= MIN_SCORE:
        return best
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        custom_list = [s.upper() for s in sys.argv[1:]]
        run_screener(custom_list)
    else:
        run_screener()
