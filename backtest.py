"""
Ravioli Backtester — runs the exact same strategy as bot_engine.py against historical data.
Usage: python backtest.py [SYMBOL] [DAYS]
  e.g. python backtest.py NVDA 30
"""

import sys
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Strategy config (mirrors bot_engine.py exactly)
# ---------------------------------------------------------------------------
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

RISK_PER_TRADE = 0.02
TAKE_PROFIT_RATIO = 3.0
ATR_SL_MULTIPLIER = 2.0
TRAILING_STOP_TRIGGER = 1.5
MAX_TRADES_PER_DAY = 4
COOLDOWN_BARS = 3

EST = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
EOD_CLOSE_TIME = dtime(15, 55)

STARTING_CAPITAL = 100_000.0

# ---------------------------------------------------------------------------
# Indicators (identical to bot_engine.py)
# ---------------------------------------------------------------------------
def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    dates = df.index.date if hasattr(df.index, 'date') else df["date"].dt.date
    cum_tp_vol = (typical * df["Volume"]).groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum()
    return cum_tp_vol / cum_vol


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = compute_ema(df["Close"], EMA_FAST)
    df["ema_slow"] = compute_ema(df["Close"], EMA_SLOW)
    df["ema_trend"] = compute_ema(df["Close"], EMA_TREND)
    df["rsi"] = compute_rsi(df["Close"], RSI_PERIOD)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )
    df["vwap"] = compute_vwap(df)
    df["atr"] = compute_atr(df, ATR_PERIOD)
    df["vol_ma"] = df["Volume"].rolling(20).mean()
    return df


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
class Backtester:
    def __init__(self, symbol: str, days: int, capital: float = STARTING_CAPITAL):
        self.symbol = symbol
        self.days = days
        self.starting_capital = capital
        self.capital = capital
        self.trades: list[dict] = []

        # Position state
        self.in_position = False
        self.position_direction = None  # "long" or "short"
        self.position_size = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_trigger = 0.0
        self.trailing_active = False

        # Daily tracking
        self.trades_today = 0
        self.current_trading_day = None
        self.bars_since_last_trade = 999

    def fetch_data(self) -> pd.DataFrame:
        end = datetime.now()
        all_frames = []
        days_left = self.days
        current_end = end

        while days_left > 0:
            chunk = min(days_left, 7)
            current_start = current_end - timedelta(days=chunk)
            print(f"  Fetching {self.symbol} data: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
            df = yf.download(self.symbol, start=current_start, end=current_end, interval="1m", progress=False)
            if not df.empty:
                all_frames.append(df)
            current_end = current_start
            days_left -= chunk

        if not all_frames:
            print(f"ERROR: No data returned for {self.symbol}. Check the symbol.")
            sys.exit(1)

        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    # -- Entry signals -----------------------------------------------------
    def check_long_entry(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < EMA_TREND + 5:
            return False
        if self.bars_since_last_trade < COOLDOWN_BARS:
            return False

        last = df.iloc[idx]
        prev = df.iloc[idx - 1]

        if not (last["Close"] > last["ema_trend"]):
            return False

        ema_bullish = last["ema_fast"] > last["ema_slow"]
        price_above_vwap = last["Close"] > last["vwap"]
        rsi_ok = 35 <= last["rsi"] <= 60

        fresh_cross = last["macd_hist"] > 0 and prev["macd_hist"] <= 0
        building = (last["macd_hist"] > 0 and prev["macd_hist"] > 0
                   and last["macd_hist"] > prev["macd_hist"])
        macd_bullish = fresh_cross or building
        volume_ok = last["Volume"] > last["vol_ma"] * 1.0 if pd.notna(last["vol_ma"]) else False

        return bool(ema_bullish and price_above_vwap and rsi_ok and macd_bullish and volume_ok)

    def check_short_entry(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < EMA_TREND + 5:
            return False
        if self.bars_since_last_trade < COOLDOWN_BARS:
            return False

        last = df.iloc[idx]
        prev = df.iloc[idx - 1]

        if not (last["Close"] < last["ema_trend"]):
            return False

        ema_bearish = last["ema_fast"] < last["ema_slow"]
        price_below_vwap = last["Close"] < last["vwap"]
        rsi_ok = 40 <= last["rsi"] <= 65

        fresh_cross = last["macd_hist"] < 0 and prev["macd_hist"] >= 0
        building = (last["macd_hist"] < 0 and prev["macd_hist"] < 0
                   and last["macd_hist"] < prev["macd_hist"])
        macd_bearish = fresh_cross or building
        volume_ok = last["Volume"] > last["vol_ma"] * 1.0 if pd.notna(last["vol_ma"]) else False

        return bool(ema_bearish and price_below_vwap and rsi_ok and macd_bearish and volume_ok)

    # -- Exit signals ------------------------------------------------------
    def check_long_exit(self, df: pd.DataFrame, idx: int) -> str | None:
        last = df.iloc[idx]
        price = last["Close"]

        if not self.trailing_active and price >= self.trailing_trigger:
            self.trailing_active = True
            self.stop_loss = self.entry_price

        if price <= self.stop_loss:
            return "TRAILING_STOP" if self.trailing_active else "STOP_LOSS"
        if price >= self.take_profit:
            return "TAKE_PROFIT"
        if last["ema_fast"] < last["ema_slow"] and price > self.entry_price:
            return "EMA_CROSSOVER"
        return None

    def check_short_exit(self, df: pd.DataFrame, idx: int) -> str | None:
        last = df.iloc[idx]
        price = last["Close"]

        # For shorts: trailing triggers when price drops enough
        if not self.trailing_active and price <= self.trailing_trigger:
            self.trailing_active = True
            self.stop_loss = self.entry_price

        # Stop loss is ABOVE entry for shorts
        if price >= self.stop_loss:
            return "TRAILING_STOP" if self.trailing_active else "STOP_LOSS"
        # Take profit is BELOW entry for shorts
        if price <= self.take_profit:
            return "TAKE_PROFIT"
        # EMA crossover — only exit when in profit (price below entry for shorts)
        if last["ema_fast"] > last["ema_slow"] and price < self.entry_price:
            return "EMA_CROSSOVER"
        return None

    # -- Main loop ---------------------------------------------------------
    def run(self):
        print(f"\n{'='*60}")
        print(f"  RAVIOLI BACKTESTER -- {self.symbol}")
        print(f"  Period: {self.days} days | Capital: ${self.starting_capital:,.2f}")
        print(f"{'='*60}\n")

        raw_df = self.fetch_data()
        print(f"  Total bars fetched: {len(raw_df)}")

        df = build_indicators(raw_df)
        print(f"  Indicators computed. Running strategy...\n")

        for i in range(EMA_TREND + 5, len(df)):
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
                self._close_position(price, ts_est, "END_OF_DAY")
                self.bars_since_last_trade = 0
                continue

            # Check exits
            if self.in_position:
                if self.position_direction == "long":
                    reason = self.check_long_exit(df, i)
                else:
                    reason = self.check_short_exit(df, i)
                if reason:
                    self._close_position(price, ts_est, reason)
                    self.bars_since_last_trade = 0
            else:
                if self.trades_today >= MAX_TRADES_PER_DAY:
                    continue
                atr = float(row["atr"]) if pd.notna(row["atr"]) else 0
                if self.check_long_entry(df, i):
                    self._open_position(price, atr, ts_est, "long")
                    self.bars_since_last_trade = 0
                elif self.check_short_entry(df, i):
                    self._open_position(price, atr, ts_est, "short")
                    self.bars_since_last_trade = 0

        if self.in_position:
            last_price = float(df.iloc[-1]["Close"])
            self._close_position(last_price, df.index[-1], "BACKTEST_END")

        self._print_results()

    def _open_position(self, price: float, atr: float, timestamp, direction: str):
        risk_amount = self.capital * RISK_PER_TRADE
        sl_distance = ATR_SL_MULTIPLIER * atr
        if sl_distance <= 0:
            return

        if direction == "long":
            self.stop_loss = round(price - sl_distance, 2)
            self.take_profit = round(price + sl_distance * TAKE_PROFIT_RATIO, 2)
            self.trailing_trigger = round(price + TRAILING_STOP_TRIGGER * atr, 2)
        else:  # short
            self.stop_loss = round(price + sl_distance, 2)
            self.take_profit = round(price - sl_distance * TAKE_PROFIT_RATIO, 2)
            self.trailing_trigger = round(price - TRAILING_STOP_TRIGGER * atr, 2)

        self.trailing_active = False

        shares = int(risk_amount / sl_distance)
        if shares <= 0:
            return
        max_shares = int(self.capital * 0.95 / price)
        shares = min(shares, max_shares)
        if shares <= 0:
            return

        self.entry_price = price
        self.position_size = shares
        self.in_position = True
        self.position_direction = direction
        self.trades_today += 1

        self.trades.append({
            "entry_time": str(timestamp),
            "entry_price": price,
            "shares": shares,
            "direction": direction,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        })

    def _close_position(self, price: float, timestamp, reason: str):
        if self.position_direction == "long":
            pnl = (price - self.entry_price) * self.position_size
        else:  # short
            pnl = (self.entry_price - price) * self.position_size

        pnl_pct = (pnl / (self.entry_price * self.position_size)) * 100
        self.capital += pnl

        self.trades[-1].update({
            "exit_time": str(timestamp),
            "exit_price": price,
            "reason": reason,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

        self.in_position = False
        self.position_direction = None
        self.position_size = 0
        self.entry_price = 0.0

    def _print_results(self):
        total_trades = len([t for t in self.trades if "pnl" in t])
        if total_trades == 0:
            print("  No trades executed during this period.")
            print("  This could mean the strategy conditions were never all met simultaneously.")
            return

        wins = [t for t in self.trades if t.get("pnl", 0) > 0]
        losses = [t for t in self.trades if t.get("pnl", 0) <= 0]
        total_pnl = sum(t.get("pnl", 0) for t in self.trades)
        total_return = ((self.capital - self.starting_capital) / self.starting_capital) * 100

        longs = [t for t in self.trades if t.get("direction") == "long" and "pnl" in t]
        shorts = [t for t in self.trades if t.get("direction") == "short" and "pnl" in t]
        long_pnl = sum(t["pnl"] for t in longs)
        short_pnl = sum(t["pnl"] for t in shorts)

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        largest_win = max(t["pnl"] for t in wins) if wins else 0
        largest_loss = min(t["pnl"] for t in losses) if losses else 0

        # Win streak / loss streak
        streaks = []
        current_streak = 0
        current_type = None
        for t in self.trades:
            if "pnl" not in t:
                continue
            is_win = t["pnl"] > 0
            if current_type == is_win:
                current_streak += 1
            else:
                if current_type is not None:
                    streaks.append((current_type, current_streak))
                current_type = is_win
                current_streak = 1
        if current_type is not None:
            streaks.append((current_type, current_streak))

        max_win_streak = max((s[1] for s in streaks if s[0]), default=0)
        max_loss_streak = max((s[1] for s in streaks if not s[0]), default=0)

        # Max drawdown
        equity_curve = [self.starting_capital]
        for t in self.trades:
            if "pnl" in t:
                equity_curve.append(equity_curve[-1] + t["pnl"])
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Exit reason breakdown
        reasons = {}
        for t in self.trades:
            r = t.get("reason", "UNKNOWN")
            reasons[r] = reasons.get(r, 0) + 1

        print(f"{'='*60}")
        print(f"  BACKTEST RESULTS -- {self.symbol}")
        print(f"{'='*60}")
        print(f"  Starting Capital:   ${self.starting_capital:>12,.2f}")
        print(f"  Final Capital:      ${self.capital:>12,.2f}")
        print(f"  Total P&L:          ${total_pnl:>12,.2f}")
        print(f"  Total Return:        {total_return:>11.2f}%")
        print(f"  Max Drawdown:        {max_dd:>11.2f}%")
        print()
        print(f"  Total Trades:        {total_trades:>8}  (L: {len(longs)} | S: {len(shorts)})")
        print(f"  Wins:                {len(wins):>8}  ({len(wins)/total_trades*100:.1f}%)")
        print(f"  Losses:              {len(losses):>8}  ({len(losses)/total_trades*100:.1f}%)")
        print(f"  Long P&L:           ${long_pnl:>12,.2f}")
        print(f"  Short P&L:          ${short_pnl:>12,.2f}")
        print()
        print(f"  Avg Win:            ${avg_win:>12,.2f}")
        print(f"  Avg Loss:           ${avg_loss:>12,.2f}")
        print(f"  Largest Win:        ${largest_win:>12,.2f}")
        print(f"  Largest Loss:       ${largest_loss:>12,.2f}")
        print(f"  Win Streak (max):    {max_win_streak:>8}")
        print(f"  Loss Streak (max):   {max_loss_streak:>8}")
        print()
        print(f"  Exit Reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<20} {count:>4}")
        print()

        # Print individual trades
        print(f"  {'-'*68}")
        print(f"  TRADE LOG")
        print(f"  {'-'*68}")
        for i, t in enumerate(self.trades, 1):
            if "pnl" not in t:
                continue
            marker = "+" if t["pnl"] > 0 else "-"
            d = "L" if t.get("direction") == "long" else "S"
            print(
                f"  {marker} #{i:<3} [{d}] {t['entry_time'][:16]}  "
                f"IN ${t['entry_price']:<8.2f} OUT ${t['exit_price']:<8.2f}  "
                f"{t['shares']:>4} sh  "
                f"P&L ${t['pnl']:>8.2f} ({t['pnl_pct']:>+.2f}%)  "
                f"[{t['reason']}]"
            )
        print(f"  {'-'*68}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    # yfinance 1-min data is limited to 30 days back
    if days > 30:
        print(f"  Note: Yahoo Finance limits 1-min data to 30 days. Capping at 30.")
        days = 30

    bt = Backtester(symbol.upper(), days)
    bt.run()
