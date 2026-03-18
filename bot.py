"""
NVDA Day Trading Bot for IBKR Paper Trading
Strategy: EMA 9/21 + RSI 14 + VWAP + MACD (12/26/9) + Volume
"""

import logging
import sys
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, Stock, util

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = "NVDA"
EXCHANGE = "SMART"
CURRENCY = "USD"
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497  # Paper trading port
CLIENT_ID = 1

# Strategy parameters
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

# Risk management
RISK_PER_TRADE = 0.02  # 2% of account
TAKE_PROFIT_RATIO = 1.5  # 1.5:1 reward-to-risk
MAX_TRADES_PER_DAY = 8
ATR_SL_MULTIPLIER = 1.2  # Tighter stop loss
TRAILING_STOP_TRIGGER = 1.0  # Trail to breakeven after 1x ATR move

# Market hours (EST)
EST = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# Bar size for live data
BAR_SIZE = "1 min"
LOOKBACK = "3 D"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = "trades.log"

logger = logging.getLogger("TradingBot")
logger.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(fmt)
logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Indicators
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
    """Cumulative VWAP reset each trading day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).groupby(df["date"].dt.date).cumsum()
    cum_vol = df["volume"].groupby(df["date"].dt.date).cumsum()
    return cum_tp_vol / cum_vol


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to the dataframe."""
    df = df.copy()
    df["ema_fast"] = compute_ema(df["close"], EMA_FAST)
    df["ema_slow"] = compute_ema(df["close"], EMA_SLOW)
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )
    df["vwap"] = compute_vwap(df)
    df["atr"] = compute_atr(df, ATR_PERIOD)
    # Volume moving average for volume confirmation
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


# ---------------------------------------------------------------------------
# Trading Bot
# ---------------------------------------------------------------------------
class TradingBot:
    def __init__(self):
        self.ib = IB()
        self.contract = Stock(SYMBOL, EXCHANGE, CURRENCY)
        self.position_size = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_trigger = 0.0  # Price level to activate trailing stop
        self.trailing_active = False
        self.in_position = False
        self.bars = None
        self.df = pd.DataFrame()
        self.last_diagnostic_time = None
        # Daily tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.daily_wins = 0
        self.daily_losses = 0
        self.current_trading_day = None
        self.daily_summary_logged = False

    # -- Connection --------------------------------------------------------
    def connect(self):
        logger.info("Connecting to TWS at %s:%s …", TWS_HOST, TWS_PORT)
        self.ib.connect(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
        self.ib.reqMarketDataType(1)  # Live data
        logger.info("Connected. Server version: %s", self.ib.client.serverVersion())

        # Verify this is a paper trading account
        self._verify_paper_account()

        # Verify connection is stable by requesting account summary
        account_value = self.get_account_value()
        if account_value <= 0:
            raise RuntimeError("Could not retrieve account value. Check TWS connection.")
        logger.info("Account verified. Net liquidation: $%.2f", account_value)

    def _verify_paper_account(self):
        """Refuse to run on a live account."""
        managed = self.ib.managedAccounts()
        if not managed:
            raise RuntimeError("No managed accounts found.")
        account_id = managed[0]
        # IBKR paper accounts start with 'D' (e.g. DU1234567)
        if not account_id.startswith("D"):
            raise RuntimeError(
                f"Account {account_id} does not appear to be a paper account. "
                "Paper accounts start with 'D'. Aborting for safety."
            )
        logger.info("Paper account confirmed: %s", account_id)

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from TWS.")

    # -- Account helpers ---------------------------------------------------
    def get_account_value(self) -> float:
        """Get net liquidation value. Checks USD first, falls back to base currency."""
        self.ib.reqAccountSummary()
        self.ib.sleep(2)
        acct = self.ib.accountSummary()
        # Prefer USD, fall back to any currency with NetLiquidation
        fallback = 0.0
        for item in acct:
            if item.tag == "NetLiquidation":
                if item.currency == "USD":
                    return float(item.value)
                if float(item.value) > 0:
                    fallback = float(item.value)
        return fallback

    def get_current_position(self) -> int:
        for pos in self.ib.positions():
            if pos.contract.symbol == SYMBOL:
                return int(pos.position)
        return 0

    # -- Data --------------------------------------------------------------
    def request_bars(self):
        """Request real-time 1-min bars and keep them updated."""
        self.bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=LOOKBACK,
            barSizeSetting=BAR_SIZE,
            whatToShow="TRADES",
            useRTH=True,
            keepUpToDate=True,
        )
        self.bars.updateEvent += self.on_bar_update
        logger.info("Subscribed to %s 1-min bars (keepUpToDate).", SYMBOL)

    def bars_to_df(self) -> pd.DataFrame:
        df = util.df(self.bars)
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    # -- Signal logic ------------------------------------------------------
    def check_entry_signal(self, df: pd.DataFrame) -> bool:
        """Return True when all entry conditions are met."""
        if len(df) < MACD_SLOW + MACD_SIGNAL + 3:
            return False

        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        ema_bullish = last["ema_fast"] > last["ema_slow"]
        price_above_vwap = last["close"] > last["vwap"]
        rsi_ok = 30 <= last["rsi"] <= 70
        # Loosened: MACD histogram positive for 2 consecutive candles
        macd_bullish = last["macd_hist"] > 0 and prev["macd_hist"] > 0
        volume_ok = last["volume"] > last["vol_ma"] * 0.8

        # Diagnostic: log to console every 5 minutes
        now = datetime.now(EST)
        if self.last_diagnostic_time is None or (now - self.last_diagnostic_time).total_seconds() >= 300:
            self.last_diagnostic_time = now
            conditions = {
                "EMA_bullish": ema_bullish,
                "Price>VWAP": price_above_vwap,
                "RSI_40-60": rsi_ok,
                "MACD_bullish": macd_bullish,
                "Volume_ok": volume_ok,
            }
            met = [k for k, v in conditions.items() if v]
            missed = [k for k, v in conditions.items() if not v]
            logger.info(
                "DIAGNOSTIC | Price=$%.2f | EMA9=%.2f EMA21=%.2f | RSI=%.1f | MACD_H=%.4f (prev=%.4f) | VWAP=%.2f | Vol=%d VolMA=%.0f",
                last["close"], last["ema_fast"], last["ema_slow"],
                last["rsi"], last["macd_hist"], prev["macd_hist"],
                last["vwap"], last["volume"], last["vol_ma"],
            )
            logger.info(
                "DIAGNOSTIC | MET: %s | MISSED: %s",
                ", ".join(met) or "none",
                ", ".join(missed) or "none",
            )

        if ema_bullish and price_above_vwap and rsi_ok and macd_bullish and volume_ok:
            logger.info(
                "ENTRY SIGNAL | EMA9=%.2f EMA21=%.2f RSI=%.1f MACD_H=%.4f VWAP=%.2f Vol=%d VolMA=%.0f",
                last["ema_fast"], last["ema_slow"], last["rsi"],
                last["macd_hist"], last["vwap"], last["volume"], last["vol_ma"],
            )
            return True
        return False

    def check_exit_signal(self, df: pd.DataFrame) -> str | None:
        """Return exit reason or None."""
        last = df.iloc[-1]
        price = last["close"]

        # Trailing stop: once price hits trigger, move stop to breakeven
        if not self.trailing_active and price >= self.trailing_trigger:
            self.trailing_active = True
            self.stop_loss = self.entry_price
            logger.info(
                "TRAILING STOP ACTIVATED | Stop moved to breakeven $%.2f",
                self.entry_price,
            )

        # EMA bearish crossover
        if last["ema_fast"] < last["ema_slow"]:
            return "EMA_CROSSOVER"

        # Stop loss (or trailing stop at breakeven)
        if price <= self.stop_loss:
            return "TRAILING_STOP" if self.trailing_active else "STOP_LOSS"

        # Take profit
        if price >= self.take_profit:
            return "TAKE_PROFIT"

        return None

    # -- Order execution ---------------------------------------------------
    def enter_long(self, price: float, atr: float):
        account_value = self.get_account_value()
        risk_amount = account_value * RISK_PER_TRADE

        sl_distance = ATR_SL_MULTIPLIER * atr
        if sl_distance <= 0:
            logger.warning("ATR-based stop distance is zero; skipping entry.")
            return

        self.stop_loss = round(price - sl_distance, 2)
        self.take_profit = round(price + sl_distance * TAKE_PROFIT_RATIO, 2)
        self.trailing_trigger = round(price + TRAILING_STOP_TRIGGER * atr, 2)
        self.trailing_active = False

        # Position size = risk / stop distance, capped to whole shares
        shares = int(risk_amount / sl_distance)
        if shares <= 0:
            logger.warning("Calculated 0 shares; skipping entry.")
            return

        # Make sure we don't exceed available capital
        max_shares = int(account_value * 0.95 / price)
        shares = min(shares, max_shares)

        order = MarketOrder("BUY", shares)
        trade = self.ib.placeOrder(self.contract, order)
        self.ib.sleep(2)  # Wait for fill

        if trade.orderStatus.status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice
            self.entry_price = fill_price
            self.position_size = shares
            self.in_position = True
            logger.info(
                "OPENED LONG | %d shares @ $%.2f | SL=$%.2f | TP=$%.2f | Risk=$%.2f (%.1f%%)",
                shares, fill_price, self.stop_loss, self.take_profit,
                risk_amount, RISK_PER_TRADE * 100,
            )
        else:
            logger.warning("Order not filled. Status: %s", trade.orderStatus.status)

    def exit_long(self, reason: str):
        if self.position_size <= 0:
            return

        order = MarketOrder("SELL", self.position_size)
        trade = self.ib.placeOrder(self.contract, order)
        self.ib.sleep(2)

        if trade.orderStatus.status == "Filled":
            exit_price = trade.orderStatus.avgFillPrice
            pnl = (exit_price - self.entry_price) * self.position_size
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
            logger.info(
                "CLOSED LONG | %s | %d shares | Entry=$%.2f | Exit=$%.2f | P&L=$%.2f (%.2f%%)",
                reason, self.position_size, self.entry_price, exit_price, pnl, pnl_pct,
            )
            self.daily_pnl += pnl
            if pnl >= 0:
                self.daily_wins += 1
            else:
                self.daily_losses += 1
            self.in_position = False
            self.position_size = 0
            self.entry_price = 0.0
            self.stop_loss = 0.0
            self.take_profit = 0.0
            self.trailing_trigger = 0.0
            self.trailing_active = False
        else:
            logger.warning("Exit order not filled. Status: %s", trade.orderStatus.status)

    # -- Real-time bar handler ---------------------------------------------
    def _reset_daily_counters(self):
        """Reset daily tracking at the start of each new trading day."""
        today = datetime.now(EST).date()
        if self.current_trading_day != today:
            if self.current_trading_day is not None:
                # Log summary for the previous day
                self._log_daily_summary()
            self.current_trading_day = today
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0
            self.daily_summary_logged = False
            logger.info("New trading day: %s", today)

    def _log_daily_summary(self):
        """Log end-of-day summary."""
        if self.daily_summary_logged:
            return
        total = self.daily_wins + self.daily_losses
        logger.info(
            "DAILY SUMMARY | Date=%s | Trades=%d | Wins=%d | Losses=%d | P&L=$%.2f",
            self.current_trading_day, total, self.daily_wins, self.daily_losses, self.daily_pnl,
        )
        self.daily_summary_logged = True

    def on_bar_update(self, bars, has_new_bar):
        if not has_new_bar:
            return

        now_est = datetime.now(EST).time()
        if not (MARKET_OPEN <= now_est <= MARKET_CLOSE):
            return

        self._reset_daily_counters()

        df = self.bars_to_df()
        if df.empty:
            return

        df = build_indicators(df)
        self.df = df
        last = df.iloc[-1]

        if self.in_position:
            reason = self.check_exit_signal(df)
            if reason:
                self.exit_long(reason)
        else:
            if self.trades_today >= MAX_TRADES_PER_DAY:
                return
            if self.check_entry_signal(df):
                self.enter_long(last["close"], last["atr"])
                self.trades_today += 1

    # -- Main loop ---------------------------------------------------------
    def run(self):
        self.connect()

        # Sync any existing position from a previous run
        existing = self.get_current_position()
        if existing > 0:
            self.in_position = True
            self.position_size = existing
            logger.info("Detected existing position: %d shares of %s", existing, SYMBOL)

        self.request_bars()
        logger.info(
            "Bot running. Trading %s during market hours (%s – %s EST). Press Ctrl+C to stop.",
            SYMBOL, MARKET_OPEN.strftime("%H:%M"), MARKET_CLOSE.strftime("%H:%M"),
        )

        try:
            while True:
                self.ib.sleep(1)

                # Auto-close at end of day if still in position
                now_est = datetime.now(EST).time()
                if now_est >= dtime(15, 55) and self.in_position:
                    logger.info("End-of-day approaching – closing position.")
                    self.exit_long("END_OF_DAY")

                # Log daily summary at market close
                if now_est >= MARKET_CLOSE and not self.daily_summary_logged:
                    self._log_daily_summary()

        except KeyboardInterrupt:
            logger.info("Shutting down …")
            if self.in_position:
                logger.info("Closing open position before exit.")
                self.exit_long("MANUAL_SHUTDOWN")
        finally:
            self.disconnect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
