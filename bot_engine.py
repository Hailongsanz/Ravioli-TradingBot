"""
Bot engine — refactored TradingBot as a controllable module with event emission.
Used by the FastAPI dashboard (app.py). The original bot.py is kept as a standalone fallback.
"""

import asyncio
import json
import logging
import threading
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from ib_insync import IB, LimitOrder, MarketOrder, Stock, util
from screener import auto_scan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = "NVDA"
EXCHANGE = "SMART"
CURRENCY = "USD"
AUTO_SCAN_TIME = dtime(9, 15)      # Run screener daily at 9:15 AM EST
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
CLIENT_ID = 1

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

EMA_TREND = 50

RISK_PER_TRADE = 0.015
TAKE_PROFIT_RATIO = 3.0
MAX_TRADES_PER_DAY = 4
ATR_SL_MULTIPLIER = 2.0
TRAILING_STOP_TRIGGER = 1.5
COOLDOWN_BARS = 3
LIMIT_ORDER_BUFFER = 0.02         # Buffer for limit orders to ensure fill

EST = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

BAR_SIZE = "1 min"
LOOKBACK = "20 D"

logger = logging.getLogger("TradingBot")


# ---------------------------------------------------------------------------
# Indicators (same as bot.py)
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
    df = df.copy()
    df["ema_fast"] = compute_ema(df["close"], EMA_FAST)
    df["ema_slow"] = compute_ema(df["close"], EMA_SLOW)
    df["ema_trend"] = compute_ema(df["close"], EMA_TREND)
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )
    df["vwap"] = compute_vwap(df)
    df["atr"] = compute_atr(df, ATR_PERIOD)
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


# ---------------------------------------------------------------------------
# Bot Engine
# ---------------------------------------------------------------------------
CONFIG_FILE = Path("ravioli_config.json")


def load_saved_symbol() -> str:
    try:
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text())
            return data.get("symbol", SYMBOL).upper()
    except Exception:
        pass
    return SYMBOL


def save_symbol(symbol: str):
    try:
        CONFIG_FILE.write_text(json.dumps({"symbol": symbol}))
    except Exception:
        pass


class BotEngine:
    def __init__(self):
        self.ib = IB()
        saved = load_saved_symbol()
        self.symbol = saved
        self.contract = Stock(saved, EXCHANGE, CURRENCY)
        self.position_size = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_trigger = 0.0
        self.trailing_active = False
        self.in_position = False
        self.position_direction = None  # "long" or "short"
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

        # Trade history for dashboard
        self.trade_history: list[dict] = []

        # Event system
        self._subscribers: list[Callable] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self.account_value = 0.0
        self.account_id = ""

        # Extended hours
        self.has_extended_hours = False
        self.market_open = MARKET_OPEN
        self.market_close = MARKET_CLOSE
        self.eod_close_time = dtime(15, 55)

        # Signal state for dashboard
        self.signal_state: dict = {}

        # Cooldown tracking
        self.bars_since_last_trade = 999

        # Auto-scan tracking
        self.last_scan_date = None
        self.auto_scan_enabled = True
        self.pending_scan = False

    def subscribe(self, callback: Callable):
        self._subscribers.append(callback)

    def _emit(self, event: dict):
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._running

    # -- Connection --------------------------------------------------------
    def connect(self):
        logger.info("Connecting to TWS at %s:%s …", TWS_HOST, TWS_PORT)
        self.ib.connect(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
        self.ib.reqMarketDataType(1)
        logger.info("Connected. Server version: %s", self.ib.client.serverVersion())
        self._verify_paper_account()
        self.account_value = self._fetch_account_value()
        if self.account_value <= 0:
            raise RuntimeError("Could not retrieve account value.")
        logger.info("Account verified. Net liquidation: $%.2f", self.account_value)

    def _verify_paper_account(self):
        managed = self.ib.managedAccounts()
        if not managed:
            raise RuntimeError("No managed accounts found.")
        self.account_id = managed[0]
        if not self.account_id.startswith("D"):
            raise RuntimeError(
                f"Account {self.account_id} is not a paper account. Aborting."
            )
        logger.info("Paper account confirmed: %s", self.account_id)

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from TWS.")

    # -- Account -----------------------------------------------------------
    def _fetch_account_value(self) -> float:
        self.ib.reqAccountSummary()
        self.ib.sleep(2)
        acct = self.ib.accountSummary()
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
            if pos.contract.symbol == self.symbol:
                return int(pos.position)
        return 0

    def set_symbol(self, symbol: str):
        """Change the trading symbol. Must be called while bot is stopped."""
        self.symbol = symbol.upper()
        self.contract = Stock(self.symbol, EXCHANGE, CURRENCY)
        self.df = pd.DataFrame()
        self.bars = None
        self.trade_history.clear()
        self.signal_state = {}
        save_symbol(self.symbol)
        logger.info("Symbol changed to %s", self.symbol)

    def switch_symbol_live(self, symbol: str) -> bool:
        """Switch symbol while staying connected to TWS. Returns True on success."""
        symbol = symbol.upper()
        if symbol == self.symbol:
            return True

        # Cancel existing bar subscription
        if self.bars is not None:
            self.ib.cancelHistoricalData(self.bars)
            self.bars = None

        self.symbol = symbol
        self.contract = Stock(self.symbol, EXCHANGE, CURRENCY)
        self.df = pd.DataFrame()
        self.trade_history.clear()
        self.signal_state = {}

        # Validate new contract
        if not self.validate_contract():
            return False

        # Subscribe to new bars
        self.request_bars()
        self._emit({"type": "status", "data": self.get_state()})
        logger.info("Switched to %s (live).", self.symbol)
        return True

    # -- Data --------------------------------------------------------------
    def validate_contract(self) -> bool:
        """Check if the contract is valid and detect extended hours."""
        details = self.ib.reqContractDetails(self.contract)
        if not details:
            logger.error("Invalid symbol: %s — not found on IBKR.", self.symbol)
            return False

        # Check trading hours for extended session
        self.has_extended_hours = False
        try:
            hours_str = details[0].tradingHours
            # IBKR format: "20260317:0400-20260317:2000;..."
            # If trading starts before 0930 or ends after 1600, extended hours exist
            if hours_str:
                for segment in hours_str.split(";"):
                    if "CLOSED" in segment:
                        continue
                    if "-" in segment:
                        start_end = segment.split("-")
                        if len(start_end) == 2:
                            start_time = start_end[0].split(":")[-1] if ":" in start_end[0] else ""
                            end_time = start_end[1].split(":")[-1] if ":" in start_end[1] else ""
                            if start_time and int(start_time) < 930:
                                self.has_extended_hours = True
                            if end_time and int(end_time) > 1600:
                                self.has_extended_hours = True
        except Exception:
            pass

        if self.has_extended_hours:
            self.market_open = dtime(4, 0)    # Pre-market 4 AM
            self.market_close = dtime(20, 0)   # After-hours 8 PM
            self.eod_close_time = dtime(19, 55)
            logger.info("%s has extended hours trading. Window: 4:00 AM - 8:00 PM EST", self.symbol)
        else:
            self.market_open = MARKET_OPEN
            self.market_close = MARKET_CLOSE
            self.eod_close_time = dtime(15, 55)
            logger.info("%s regular hours only. Window: 9:30 AM - 4:00 PM EST", self.symbol)

        return True

    def request_bars(self):
        self.bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=LOOKBACK,
            barSizeSetting=BAR_SIZE,
            whatToShow="TRADES",
            useRTH=not self.has_extended_hours,
            keepUpToDate=True,
        )
        self.bars.updateEvent += self.on_bar_update
        logger.info("Subscribed to %s 1-min bars.", self.symbol)
        self._emit({"type": "bars_ready"})

    def bars_to_df(self) -> pd.DataFrame:
        df = util.df(self.bars)
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    def get_bars_snapshot(self) -> list[dict]:
        """Return all bars with indicators for initial chart load."""
        if self.df.empty:
            raw = self.bars_to_df()
            if raw.empty:
                return []
            self.df = build_indicators(raw)
        records = []
        for _, row in self.df.iterrows():
            records.append({
                "time": int(row["date"].timestamp()) if hasattr(row["date"], "timestamp") else 0,
                "open": round(row["open"], 2),
                "high": round(row["high"], 2),
                "low": round(row["low"], 2),
                "close": round(row["close"], 2),
                "volume": int(row["volume"]),
                "ema_fast": round(row["ema_fast"], 2) if pd.notna(row["ema_fast"]) else None,
                "ema_slow": round(row["ema_slow"], 2) if pd.notna(row["ema_slow"]) else None,
                "ema_trend": round(row["ema_trend"], 2) if pd.notna(row["ema_trend"]) else None,
                "rsi": round(row["rsi"], 2) if pd.notna(row["rsi"]) else None,
                "macd": round(row["macd"], 4) if pd.notna(row["macd"]) else None,
                "macd_signal": round(row["macd_signal"], 4) if pd.notna(row["macd_signal"]) else None,
                "macd_hist": round(row["macd_hist"], 4) if pd.notna(row["macd_hist"]) else None,
                "vwap": round(row["vwap"], 2) if pd.notna(row["vwap"]) else None,
                "atr": round(row["atr"], 4) if pd.notna(row["atr"]) else None,
            })
        return records

    def get_state(self) -> dict:
        """Full state snapshot for the dashboard."""
        return {
            "running": self._running,
            "connected": self.ib.isConnected(),
            "account_id": self.account_id,
            "account_value": self.account_value,
            "symbol": self.symbol,
            "in_position": self.in_position,
            "position_direction": self.position_direction,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_active": self.trailing_active,
            "extended_hours": self.has_extended_hours,
            "market_open": self.market_open.strftime("%H:%M"),
            "market_close": self.market_close.strftime("%H:%M"),
            "trades_today": self.trades_today,
            "max_trades": MAX_TRADES_PER_DAY,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
            "signal_state": self.signal_state,
            "auto_scan_enabled": self.auto_scan_enabled,
            "last_scan_date": str(self.last_scan_date) if self.last_scan_date else None,
        }

    # -- Signal logic ------------------------------------------------------
    def check_long_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < EMA_TREND + 5:
            return False
        if self.bars_since_last_trade < COOLDOWN_BARS:
            return False

        last = df.iloc[-1]
        prev = df.iloc[-2]

        trend_bullish = bool(last["close"] > last["ema_trend"])
        ema_bullish = bool(last["ema_fast"] > last["ema_slow"])
        price_above_vwap = bool(last["close"] > last["vwap"])
        rsi_ok = bool(35 <= last["rsi"] <= 60)

        fresh_cross = bool(last["macd_hist"] > 0 and prev["macd_hist"] <= 0)
        building = bool(
            last["macd_hist"] > 0 and prev["macd_hist"] > 0
            and last["macd_hist"] > prev["macd_hist"]
        )
        macd_bullish = fresh_cross or building
        volume_ok = bool(last["volume"] > last["vol_ma"] * 1.0) if pd.notna(last["vol_ma"]) else False

        return bool(trend_bullish and ema_bullish and price_above_vwap and rsi_ok and macd_bullish and volume_ok)

    def check_short_entry(self, df: pd.DataFrame) -> bool:
        if len(df) < EMA_TREND + 5:
            return False
        if self.bars_since_last_trade < COOLDOWN_BARS:
            return False

        last = df.iloc[-1]
        prev = df.iloc[-2]

        trend_bearish = bool(last["close"] < last["ema_trend"])
        ema_bearish = bool(last["ema_fast"] < last["ema_slow"])
        price_below_vwap = bool(last["close"] < last["vwap"])
        rsi_ok = bool(40 <= last["rsi"] <= 65)

        # MACD crossing negative or building downward momentum
        fresh_cross = bool(last["macd_hist"] < 0 and prev["macd_hist"] >= 0)
        building = bool(
            last["macd_hist"] < 0 and prev["macd_hist"] < 0
            and last["macd_hist"] < prev["macd_hist"]
        )
        macd_bearish = fresh_cross or building
        volume_ok = bool(last["volume"] > last["vol_ma"] * 1.0) if pd.notna(last["vol_ma"]) else False

        return bool(trend_bearish and ema_bearish and price_below_vwap and rsi_ok and macd_bearish and volume_ok)

    def update_signal_state(self, df: pd.DataFrame):
        """Update signal state for dashboard display."""
        if len(df) < EMA_TREND + 5:
            return
        last = df.iloc[-1]
        prev = df.iloc[-2]

        self.signal_state = {
            "trend_bullish": bool(last["close"] > last["ema_trend"]),
            "ema_bullish": bool(last["ema_fast"] > last["ema_slow"]),
            "price_above_vwap": bool(last["close"] > last["vwap"]),
            "rsi_ok": bool(35 <= last["rsi"] <= 65),
            "macd_bullish": bool(last["macd_hist"] > 0),
            "volume_ok": bool(last["volume"] > last["vol_ma"] * 1.0) if pd.notna(last["vol_ma"]) else False,
            "rsi_value": round(float(last["rsi"]), 1),
            "macd_hist_value": round(float(last["macd_hist"]), 4),
            "macd_hist_prev": round(float(prev["macd_hist"]), 4),
            "cooldown": self.bars_since_last_trade < COOLDOWN_BARS,
        }
        self._emit({"type": "signal", "data": self.signal_state})

    def check_long_exit(self, df: pd.DataFrame) -> str | None:
        last = df.iloc[-1]
        price = last["close"]

        # Trailing stop: once price reaches 1.5x ATR profit, move stop to breakeven
        if not self.trailing_active and self.trailing_trigger > 0:
            if price >= self.trailing_trigger:
                self.trailing_active = True
                self.stop_loss = self.entry_price
                logger.info("TRAILING STOP activated (LONG) -> breakeven $%.2f", self.stop_loss)

        if price <= self.stop_loss:
            return "TRAILING_STOP" if self.trailing_active else "STOP_LOSS"
        if price >= self.take_profit:
            return "TAKE_PROFIT"
        if last["ema_fast"] < last["ema_slow"] and price > self.entry_price:
            return "EMA_CROSSOVER"
        return None

    def check_short_exit(self, df: pd.DataFrame) -> str | None:
        last = df.iloc[-1]
        price = last["close"]

        # Trailing stop: once price drops 1.5x ATR below entry, move stop to breakeven
        if not self.trailing_active and self.trailing_trigger > 0:
            if price <= self.trailing_trigger:
                self.trailing_active = True
                self.stop_loss = self.entry_price
                logger.info("TRAILING STOP activated (SHORT) -> breakeven $%.2f", self.stop_loss)

        # Stop loss is ABOVE entry for shorts
        if price >= self.stop_loss:
            return "TRAILING_STOP" if self.trailing_active else "STOP_LOSS"
        # Take profit is BELOW entry for shorts
        if price <= self.take_profit:
            return "TAKE_PROFIT"
        # EMA crossover exit — only when in profit (price below entry for shorts)
        if last["ema_fast"] > last["ema_slow"] and price < self.entry_price:
            return "EMA_CROSSOVER"
        return None

    # -- Order execution ---------------------------------------------------
    def _reset_position(self):
        self.in_position = False
        self.position_direction = None
        self.position_size = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_trigger = 0.0
        self.trailing_active = False

    def enter_long(self, price: float, atr: float):
        self.account_value = self._fetch_account_value()
        risk_amount = self.account_value * RISK_PER_TRADE

        sl_distance = ATR_SL_MULTIPLIER * atr
        if sl_distance <= 0:
            return

        self.stop_loss = round(price - sl_distance, 2)
        self.take_profit = round(price + sl_distance * TAKE_PROFIT_RATIO, 2)
        self.trailing_trigger = round(price + TRAILING_STOP_TRIGGER * atr, 2)

        shares = int(risk_amount / sl_distance)
        if shares <= 0:
            return
        max_shares = int(self.account_value * 0.95 / price)
        shares = min(shares, max_shares)

        # Limit order with small buffer for price protection
        limit_price = round(price + LIMIT_ORDER_BUFFER, 2)
        order = LimitOrder("BUY", shares, limit_price)
        trade = self.ib.placeOrder(self.contract, order)
        self.ib.sleep(3)

        if trade.orderStatus.status != "Filled":
            self.ib.sleep(2)
        if trade.orderStatus.status != "Filled":
            self.ib.cancelOrder(order)
            logger.warning("Long limit order not filled — cancelled.")
            return

        fill_price = trade.orderStatus.avgFillPrice
        self.entry_price = fill_price
        self.position_size = shares
        self.in_position = True
        self.position_direction = "long"

        trade_record = {
            "time": datetime.now(EST).isoformat(),
            "side": "BUY",
            "direction": "long",
            "shares": shares,
            "price": fill_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }
        self.trade_history.append(trade_record)
        self._emit({"type": "trade", "data": trade_record})
        self._emit({"type": "status", "data": self.get_state()})

        logger.info(
            "OPENED LONG | %d shares @ $%.2f | SL=$%.2f | TP=$%.2f",
            shares, fill_price, self.stop_loss, self.take_profit,
        )

    def exit_long(self, reason: str):
        if self.position_size <= 0:
            return

        limit_price = round(self.df.iloc[-1]["close"] - LIMIT_ORDER_BUFFER, 2) if not self.df.empty else 0
        if limit_price > 0:
            order = LimitOrder("SELL", self.position_size, limit_price)
        else:
            order = MarketOrder("SELL", self.position_size)
        trade = self.ib.placeOrder(self.contract, order)
        self.ib.sleep(3)

        # Fallback to market if limit didn't fill
        if trade.orderStatus.status != "Filled":
            self.ib.sleep(2)
        if trade.orderStatus.status != "Filled" and limit_price > 0:
            self.ib.cancelOrder(order)
            order = MarketOrder("SELL", self.position_size)
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)

        if trade.orderStatus.status == "Filled":
            exit_price = trade.orderStatus.avgFillPrice
            pnl = (exit_price - self.entry_price) * self.position_size
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100

            trade_record = {
                "time": datetime.now(EST).isoformat(),
                "side": "SELL",
                "direction": "long",
                "reason": reason,
                "shares": self.position_size,
                "price": exit_price,
                "entry_price": self.entry_price,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            }
            self.trade_history.append(trade_record)
            self._emit({"type": "trade", "data": trade_record})

            self.daily_pnl += pnl
            if pnl >= 0:
                self.daily_wins += 1
            else:
                self.daily_losses += 1

            logger.info(
                "CLOSED LONG | %s | %d shares | Entry=$%.2f | Exit=$%.2f | P&L=$%.2f (%.2f%%)",
                reason, self.position_size, self.entry_price, exit_price, pnl, pnl_pct,
            )
            self._reset_position()
            self._emit({"type": "status", "data": self.get_state()})
        else:
            logger.warning("Exit long not filled. Status: %s", trade.orderStatus.status)

    def enter_short(self, price: float, atr: float):
        self.account_value = self._fetch_account_value()
        risk_amount = self.account_value * RISK_PER_TRADE

        sl_distance = ATR_SL_MULTIPLIER * atr
        if sl_distance <= 0:
            return

        # For shorts: stop is ABOVE entry, take profit BELOW
        self.stop_loss = round(price + sl_distance, 2)
        self.take_profit = round(price - sl_distance * TAKE_PROFIT_RATIO, 2)
        self.trailing_trigger = round(price - TRAILING_STOP_TRIGGER * atr, 2)

        shares = int(risk_amount / sl_distance)
        if shares <= 0:
            return
        max_shares = int(self.account_value * 0.95 / price)
        shares = min(shares, max_shares)

        # Limit order with small buffer for price protection
        limit_price = round(price - LIMIT_ORDER_BUFFER, 2)
        order = LimitOrder("SELL", shares, limit_price)
        trade = self.ib.placeOrder(self.contract, order)
        self.ib.sleep(3)

        if trade.orderStatus.status != "Filled":
            self.ib.sleep(2)
        if trade.orderStatus.status != "Filled":
            self.ib.cancelOrder(order)
            logger.warning("Short limit order not filled — cancelled.")
            return

        fill_price = trade.orderStatus.avgFillPrice
        self.entry_price = fill_price
        self.position_size = shares
        self.in_position = True
        self.position_direction = "short"

        trade_record = {
            "time": datetime.now(EST).isoformat(),
            "side": "SELL",
            "direction": "short",
            "shares": shares,
            "price": fill_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }
        self.trade_history.append(trade_record)
        self._emit({"type": "trade", "data": trade_record})
        self._emit({"type": "status", "data": self.get_state()})

        logger.info(
            "OPENED SHORT | %d shares @ $%.2f | SL=$%.2f | TP=$%.2f",
            shares, fill_price, self.stop_loss, self.take_profit,
        )

    def exit_short(self, reason: str):
        if self.position_size <= 0:
            return

        limit_price = round(self.df.iloc[-1]["close"] + LIMIT_ORDER_BUFFER, 2) if not self.df.empty else 0
        if limit_price > 0:
            order = LimitOrder("BUY", self.position_size, limit_price)
        else:
            order = MarketOrder("BUY", self.position_size)
        trade = self.ib.placeOrder(self.contract, order)
        self.ib.sleep(3)

        # Fallback to market if limit didn't fill
        if trade.orderStatus.status != "Filled":
            self.ib.sleep(2)
        if trade.orderStatus.status != "Filled" and limit_price > 0:
            self.ib.cancelOrder(order)
            order = MarketOrder("BUY", self.position_size)
            trade = self.ib.placeOrder(self.contract, order)
            self.ib.sleep(2)

        if trade.orderStatus.status == "Filled":
            exit_price = trade.orderStatus.avgFillPrice
            # Short P&L: profit when price goes DOWN
            pnl = (self.entry_price - exit_price) * self.position_size
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100

            trade_record = {
                "time": datetime.now(EST).isoformat(),
                "side": "BUY",
                "direction": "short",
                "reason": reason,
                "shares": self.position_size,
                "price": exit_price,
                "entry_price": self.entry_price,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            }
            self.trade_history.append(trade_record)
            self._emit({"type": "trade", "data": trade_record})

            self.daily_pnl += pnl
            if pnl >= 0:
                self.daily_wins += 1
            else:
                self.daily_losses += 1

            logger.info(
                "CLOSED SHORT | %s | %d shares | Entry=$%.2f | Exit=$%.2f | P&L=$%.2f (%.2f%%)",
                reason, self.position_size, self.entry_price, exit_price, pnl, pnl_pct,
            )
            self._reset_position()
            self._emit({"type": "status", "data": self.get_state()})
        else:
            logger.warning("Exit short not filled. Status: %s", trade.orderStatus.status)

    # -- Auto-scan ---------------------------------------------------------
    def run_auto_scan(self):
        """Run the screener and switch to the best ticker if it qualifies."""
        try:
            logger.info("AUTO-SCAN: Running daily screener...")
            self._emit({"type": "scan", "data": {"status": "scanning"}})

            best = auto_scan()
            if best is None:
                logger.info("AUTO-SCAN: No ticker scored above threshold. Keeping %s.", self.symbol)
                self._emit({"type": "scan", "data": {
                    "status": "no_pick",
                    "message": "No ticker above threshold",
                    "current": self.symbol,
                }})
                return

            if best["symbol"] == self.symbol:
                logger.info("AUTO-SCAN: %s still the best pick (score: %.1f). No switch needed.",
                            self.symbol, best["score"])
                self._emit({"type": "scan", "data": {
                    "status": "same",
                    "symbol": best["symbol"],
                    "score": best["score"],
                }})
                return

            old_symbol = self.symbol
            logger.info("AUTO-SCAN: Switching %s -> %s (score: %.1f)",
                        old_symbol, best["symbol"], best["score"])

            if self.switch_symbol_live(best["symbol"]):
                save_symbol(best["symbol"])
                self._emit({"type": "scan", "data": {
                    "status": "switched",
                    "from": old_symbol,
                    "to": best["symbol"],
                    "score": best["score"],
                }})
                logger.info("AUTO-SCAN: Now trading %s (score: %.1f, price: $%.2f)",
                            best["symbol"], best["score"], best.get("price", 0))
            else:
                logger.error("AUTO-SCAN: Failed to switch to %s. Keeping %s.",
                             best["symbol"], old_symbol)
        except Exception as e:
            logger.error("AUTO-SCAN error: %s", e)

    # -- Bar handler -------------------------------------------------------
    def _reset_daily_counters(self):
        today = datetime.now(EST).date()
        if self.current_trading_day != today:
            if self.current_trading_day is not None:
                self._log_daily_summary()
            self.current_trading_day = today
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0
            self.daily_summary_logged = False
            logger.info("New trading day: %s", today)

    def _log_daily_summary(self):
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

        now_est = datetime.now(EST)

        df = self.bars_to_df()
        if df.empty:
            return

        df = build_indicators(df)
        self.df = df
        last = df.iloc[-1]

        # Emit bar event for chart
        bar_data = {
            "time": int(last["date"].timestamp()) if hasattr(last["date"], "timestamp") else 0,
            "open": round(float(last["open"]), 2),
            "high": round(float(last["high"]), 2),
            "low": round(float(last["low"]), 2),
            "close": round(float(last["close"]), 2),
            "volume": int(last["volume"]),
            "ema_fast": round(float(last["ema_fast"]), 2) if pd.notna(last["ema_fast"]) else None,
            "ema_slow": round(float(last["ema_slow"]), 2) if pd.notna(last["ema_slow"]) else None,
            "ema_trend": round(float(last["ema_trend"]), 2) if pd.notna(last["ema_trend"]) else None,
            "rsi": round(float(last["rsi"]), 2) if pd.notna(last["rsi"]) else None,
            "macd": round(float(last["macd"]), 4) if pd.notna(last["macd"]) else None,
            "macd_signal": round(float(last["macd_signal"]), 4) if pd.notna(last["macd_signal"]) else None,
            "macd_hist": round(float(last["macd_hist"]), 4) if pd.notna(last["macd_hist"]) else None,
            "vwap": round(float(last["vwap"]), 2) if pd.notna(last["vwap"]) else None,
        }
        self._emit({"type": "bar", "data": bar_data})

        if not (self.market_open <= now_est.time() <= self.market_close):
            return

        self._reset_daily_counters()
        self.bars_since_last_trade += 1

        if self.in_position:
            if self.position_direction == "long":
                reason = self.check_long_exit(df)
                if reason:
                    self.exit_long(reason)
                    self.bars_since_last_trade = 0
            elif self.position_direction == "short":
                reason = self.check_short_exit(df)
                if reason:
                    self.exit_short(reason)
                    self.bars_since_last_trade = 0
        else:
            if self.trades_today >= MAX_TRADES_PER_DAY:
                self.update_signal_state(df)
                return
            if self.check_long_entry(df):
                self.enter_long(last["close"], last["atr"])
                self.trades_today += 1
                self.bars_since_last_trade = 0
            elif self.check_short_entry(df):
                self.enter_short(last["close"], last["atr"])
                self.trades_today += 1
                self.bars_since_last_trade = 0
            self.update_signal_state(df)

        # Emit account update
        self._emit({"type": "account", "data": {
            "account_value": self.account_value,
            "daily_pnl": round(self.daily_pnl, 2),
            "trades_today": self.trades_today,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses,
        }})

    # -- Lifecycle ---------------------------------------------------------
    async def start(self):
        if self._running:
            return {"status": "already_running"}
        # Run IB on a dedicated thread with its own event loop
        self._running = True
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        # Wait briefly for connection
        for _ in range(30):
            await asyncio.sleep(0.5)
            if self.ib.isConnected():
                break
        if not self.ib.isConnected():
            self._running = False
            raise RuntimeError("Failed to connect to TWS.")
        self._emit({"type": "status", "data": self.get_state()})
        logger.info("Bot started.")
        return {"status": "started"}

    def _run_thread(self):
        """Runs on a background thread — owns the IB event loop."""
        import time
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            time.sleep(2)  # Brief pause to avoid TWS rate limiting
            self.connect()
            if not self.validate_contract():
                self._emit({"type": "error", "data": {"message": f"Invalid symbol: {self.symbol}"}})
                return
            existing = self.get_current_position()
            if existing > 0:
                self.in_position = True
                self.position_size = existing
                self.position_direction = "long"
                logger.info("Detected existing LONG position: %d shares", existing)
            elif existing < 0:
                self.in_position = True
                self.position_size = abs(existing)
                self.position_direction = "short"
                logger.info("Detected existing SHORT position: %d shares", abs(existing))
            self.request_bars()

            # Emit full state now that account, position, and bars are ready
            self._emit({"type": "status", "data": self.get_state()})

            while self._running:
                self.ib.sleep(1)
                now_est = datetime.now(EST)
                now_time = now_est.time()
                today = now_est.date()

                # Auto-scan at 9:15 AM — pick best ticker before market opens
                if (self.auto_scan_enabled
                        and self.last_scan_date != today
                        and now_time >= AUTO_SCAN_TIME
                        and now_time < MARKET_OPEN
                        and not self.in_position):
                    self.last_scan_date = today
                    self.run_auto_scan()

                # Manual scan triggered from UI
                if self.pending_scan:
                    self.pending_scan = False
                    self.run_auto_scan()

                if now_time >= self.eod_close_time and self.in_position:
                    logger.info("End-of-day — closing position.")
                    if self.position_direction == "short":
                        self.exit_short("END_OF_DAY")
                    else:
                        self.exit_long("END_OF_DAY")
                if now_time >= self.market_close and not self.daily_summary_logged:
                    self._log_daily_summary()
        except Exception as e:
            logger.error("Bot thread error: %s", e)
        finally:
            self.disconnect()
            self._running = False
            self._emit({"type": "status", "data": self.get_state()})

    async def stop(self):
        if not self._running:
            return {"status": "not_running"}
        self._running = False
        if self.in_position:
            if self.position_direction == "short":
                self.exit_short("MANUAL_SHUTDOWN")
            else:
                self.exit_long("MANUAL_SHUTDOWN")
        if self._thread:
            self._thread.join(timeout=10)
        self._emit({"type": "status", "data": self.get_state()})
        logger.info("Bot stopped.")
        return {"status": "stopped"}
