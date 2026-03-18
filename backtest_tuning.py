"""
Ravioli Strategy Tuning — test improved parameters against historical data.
"""

import sys
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

EST = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
EOD_CLOSE_TIME = dtime(15, 55)

STARTING_CAPITAL = 100_000.0


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast, slow, signal):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_vwap(df):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    dates = df.index.date
    cum_tp_vol = (typical * df["Volume"]).groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum()
    return cum_tp_vol / cum_vol

def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_indicators(df, ema_fast=9, ema_slow=21, ema_trend=50):
    df = df.copy()
    df["ema_fast"] = compute_ema(df["Close"], ema_fast)
    df["ema_slow"] = compute_ema(df["Close"], ema_slow)
    df["ema_trend"] = compute_ema(df["Close"], ema_trend)
    df["rsi"] = compute_rsi(df["Close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["Close"], 12, 26, 9)
    df["vwap"] = compute_vwap(df)
    df["atr"] = compute_atr(df, 14)
    df["vol_ma"] = df["Volume"].rolling(20).mean()
    return df


# ---------------------------------------------------------------------------
# Strategy configs to test
# ---------------------------------------------------------------------------
STRATEGIES = {
    "CURRENT (baseline)": {
        "ema_fast": 9, "ema_slow": 21,
        "use_trend_filter": False,
        "rsi_low": 30, "rsi_high": 70,
        "macd_mode": "positive",      # just histogram > 0
        "volume_mult": 0.8,
        "atr_sl_mult": 1.2,
        "tp_ratio": 1.5,
        "trailing_trigger_atr": 1.0,
        "max_trades": 8,
        "cooldown_bars": 0,
    },
    "V2 - Quality Entries": {
        "ema_fast": 9, "ema_slow": 21,
        "use_trend_filter": True,       # price > EMA 50
        "rsi_low": 40, "rsi_high": 65,  # tighter RSI band
        "macd_mode": "crossover",       # histogram crosses from - to +
        "volume_mult": 1.2,             # stronger volume required
        "atr_sl_mult": 1.8,             # wider stops
        "tp_ratio": 2.5,               # better R:R
        "trailing_trigger_atr": 1.5,
        "max_trades": 4,
        "cooldown_bars": 3,
    },
    "V3 - Trend Rider": {
        "ema_fast": 9, "ema_slow": 21,
        "use_trend_filter": True,
        "rsi_low": 35, "rsi_high": 60,
        "macd_mode": "crossover",
        "volume_mult": 1.0,
        "atr_sl_mult": 2.0,             # very wide stops
        "tp_ratio": 3.0,               # big winners
        "trailing_trigger_atr": 1.5,
        "max_trades": 3,
        "cooldown_bars": 5,
    },
    "V4 - Balanced": {
        "ema_fast": 9, "ema_slow": 21,
        "use_trend_filter": True,
        "rsi_low": 40, "rsi_high": 65,
        "macd_mode": "crossover",
        "volume_mult": 1.0,
        "atr_sl_mult": 1.5,
        "tp_ratio": 2.0,
        "trailing_trigger_atr": 1.2,
        "max_trades": 5,
        "cooldown_bars": 3,
    },
}


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
class Backtester:
    def __init__(self, config: dict, capital=STARTING_CAPITAL):
        self.cfg = config
        self.starting_capital = capital
        self.capital = capital
        self.trades = []
        self.in_position = False
        self.position_size = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_trigger = 0.0
        self.trailing_active = False
        self.trades_today = 0
        self.current_trading_day = None
        self.bars_since_last_trade = 999

    def check_entry(self, df, idx):
        cfg = self.cfg
        if idx < 52:
            return False

        last = df.iloc[idx]
        prev = df.iloc[idx - 1]

        # Cooldown
        if self.bars_since_last_trade < cfg["cooldown_bars"]:
            return False

        # Trend filter: price above EMA 50
        if cfg["use_trend_filter"]:
            if last["Close"] < last["ema_trend"]:
                return False

        # EMA alignment
        if not (last["ema_fast"] > last["ema_slow"]):
            return False

        # Price above VWAP
        if not (last["Close"] > last["vwap"]):
            return False

        # RSI in range
        if not (cfg["rsi_low"] <= last["rsi"] <= cfg["rsi_high"]):
            return False

        # MACD condition
        if cfg["macd_mode"] == "positive":
            if not (last["macd_hist"] > 0 and prev["macd_hist"] > 0):
                return False
        elif cfg["macd_mode"] == "crossover":
            # Fresh crossover: current > 0, previous <= 0
            # OR strong momentum: current > 0 and increasing for 2 bars
            fresh_cross = last["macd_hist"] > 0 and prev["macd_hist"] <= 0
            building = (last["macd_hist"] > 0 and prev["macd_hist"] > 0
                       and last["macd_hist"] > prev["macd_hist"])
            if not (fresh_cross or building):
                return False

        # Volume
        if pd.notna(last["vol_ma"]) and last["vol_ma"] > 0:
            if not (last["Volume"] > last["vol_ma"] * cfg["volume_mult"]):
                return False

        return True

    def check_exit(self, df, idx):
        cfg = self.cfg
        last = df.iloc[idx]
        price = last["Close"]

        # Trailing stop activation
        if not self.trailing_active and price >= self.trailing_trigger:
            self.trailing_active = True
            self.stop_loss = self.entry_price

        # Stop loss
        if price <= self.stop_loss:
            return "TRAILING_STOP" if self.trailing_active else "STOP_LOSS"

        # Take profit
        if price >= self.take_profit:
            return "TAKE_PROFIT"

        # EMA crossover exit — only if we're in profit (don't cut losers short on weak signal)
        if last["ema_fast"] < last["ema_slow"] and price > self.entry_price:
            return "EMA_CROSSOVER"

        return None

    def run(self, df):
        self.trades = []
        self.capital = self.starting_capital
        self.in_position = False

        for i in range(52, len(df)):
            row = df.iloc[i]
            ts = df.index[i]
            ts_est = ts.astimezone(EST) if ts.tzinfo else ts
            bar_time = ts_est.time()
            bar_date = ts_est.date()

            if not (MARKET_OPEN <= bar_time <= MARKET_CLOSE):
                continue

            if self.current_trading_day != bar_date:
                self.current_trading_day = bar_date
                self.trades_today = 0

            self.bars_since_last_trade += 1
            price = float(row["Close"])

            # EOD close
            if bar_time >= EOD_CLOSE_TIME and self.in_position:
                self._close(price, ts_est, "END_OF_DAY")
                continue

            if self.in_position:
                reason = self.check_exit(df, i)
                if reason:
                    self._close(price, ts_est, reason)
            else:
                if self.trades_today >= self.cfg["max_trades"]:
                    continue
                if self.check_entry(df, i):
                    atr = float(row["atr"]) if pd.notna(row["atr"]) else 0
                    self._open(price, atr, ts_est)

        if self.in_position:
            self._close(float(df.iloc[-1]["Close"]), df.index[-1], "BACKTEST_END")

        return self._results()

    def _open(self, price, atr, ts):
        cfg = self.cfg
        risk = self.capital * 0.02
        sl_dist = cfg["atr_sl_mult"] * atr
        if sl_dist <= 0:
            return

        shares = int(risk / sl_dist)
        max_shares = int(self.capital * 0.95 / price)
        shares = min(shares, max_shares)
        if shares <= 0:
            return

        self.entry_price = price
        self.position_size = shares
        self.stop_loss = round(price - sl_dist, 2)
        self.take_profit = round(price + sl_dist * cfg["tp_ratio"], 2)
        self.trailing_trigger = round(price + cfg["trailing_trigger_atr"] * atr, 2)
        self.trailing_active = False
        self.in_position = True
        self.trades_today += 1
        self.bars_since_last_trade = 0

        self.trades.append({
            "entry_time": str(ts), "entry_price": price, "shares": shares,
            "stop_loss": self.stop_loss, "take_profit": self.take_profit,
        })

    def _close(self, price, ts, reason):
        pnl = (price - self.entry_price) * self.position_size
        self.capital += pnl
        self.trades[-1].update({
            "exit_time": str(ts), "exit_price": price, "reason": reason,
            "pnl": round(pnl, 2), "pnl_pct": round((price - self.entry_price) / self.entry_price * 100, 2),
        })
        self.in_position = False
        self.position_size = 0
        self.bars_since_last_trade = 0

    def _results(self):
        completed = [t for t in self.trades if "pnl" in t]
        if not completed:
            return {"trades": 0, "pnl": 0, "win_rate": 0, "max_dd": 0}

        wins = [t for t in completed if t["pnl"] > 0]
        losses = [t for t in completed if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in completed)

        # Max drawdown
        equity = [self.starting_capital]
        for t in completed:
            equity.append(equity[-1] + t["pnl"])
        peak = equity[0]
        max_dd = 0
        for eq in equity:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd: max_dd = dd

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1

        # Exit reasons
        reasons = {}
        for t in completed:
            r = t.get("reason", "?")
            reasons[r] = reasons.get(r, 0) + 1

        return {
            "trades": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(completed) * 100 if completed else 0,
            "pnl": total_pnl,
            "return_pct": (self.capital - self.starting_capital) / self.starting_capital * 100,
            "max_dd": max_dd,
            "avg_win": np.mean([t["pnl"] for t in wins]) if wins else 0,
            "avg_loss": np.mean([t["pnl"] for t in losses]) if losses else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
            "reasons": reasons,
        }


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_data(symbol, days):
    end = datetime.now()
    all_frames = []
    days_left = days
    current_end = end
    while days_left > 0:
        chunk = min(days_left, 7)
        current_start = current_end - timedelta(days=chunk)
        df = yf.download(symbol, start=current_start, end=current_end, interval="1m", progress=False)
        if not df.empty:
            all_frames.append(df)
        current_end = current_start
        days_left -= chunk

    if not all_frames:
        print(f"  No data for {symbol}")
        return pd.DataFrame()

    df = pd.concat(all_frames).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    symbols = ["NVDA", "AAPL", "TSLA"]
    days = 30

    print(f"\n{'='*80}")
    print(f"  RAVIOLI STRATEGY COMPARISON")
    print(f"  Symbols: {', '.join(symbols)} | Period: {days} days | Capital: ${STARTING_CAPITAL:,.0f}")
    print(f"{'='*80}\n")

    # Fetch all data first
    data = {}
    for sym in symbols:
        print(f"  Fetching {sym}...")
        raw = fetch_data(sym, days)
        if not raw.empty:
            data[sym] = build_indicators(raw)
            print(f"    {len(raw)} bars loaded")

    print(f"\n{'='*80}")

    # Test each strategy
    for name, config in STRATEGIES.items():
        print(f"\n  >> {name}")
        print(f"  {'  '}{'-'*60}")

        all_results = []
        for sym in symbols:
            if sym not in data:
                continue
            bt = Backtester(config)
            result = bt.run(data[sym])
            all_results.append(result)
            print(
                f"    {sym:5}  "
                f"Trades: {result['trades']:>4}  "
                f"WR: {result['win_rate']:>5.1f}%  "
                f"P&L: ${result['pnl']:>9,.2f}  "
                f"Return: {result['return_pct']:>+6.2f}%  "
                f"MaxDD: {result['max_dd']:>5.2f}%  "
                f"PF: {result['profit_factor']:>4.2f}"
            )

        # Aggregate
        if all_results:
            total_trades = sum(r["trades"] for r in all_results)
            total_pnl = sum(r["pnl"] for r in all_results)
            avg_wr = np.mean([r["win_rate"] for r in all_results if r["trades"] > 0])
            avg_dd = np.mean([r["max_dd"] for r in all_results if r["trades"] > 0])
            avg_pf = np.mean([r["profit_factor"] for r in all_results if r["trades"] > 0])

            print(f"  {'  '}{'-'*60}")
            print(
                f"    {'AVG':5}  "
                f"Trades: {total_trades:>4}  "
                f"WR: {avg_wr:>5.1f}%  "
                f"P&L: ${total_pnl:>9,.2f}  "
                f"Return: {total_pnl/STARTING_CAPITAL/len(symbols)*100:>+6.2f}%  "
                f"MaxDD: {avg_dd:>5.2f}%  "
                f"PF: {avg_pf:>4.2f}"
            )

        # Print exit reason breakdown for this strategy
        if all_results:
            merged_reasons = {}
            for r in all_results:
                for reason, count in r.get("reasons", {}).items():
                    merged_reasons[reason] = merged_reasons.get(reason, 0) + count
            reason_str = "  |  ".join(f"{k}: {v}" for k, v in sorted(merged_reasons.items(), key=lambda x: -x[1]))
            print(f"    Exits: {reason_str}")

    print(f"\n{'='*80}")
    print()
