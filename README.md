# Ravioli Trading Bot

Ravioli is an automated day trading bot that connects to Interactive Brokers (IBKR) via TWS/Gateway. It trades 1-minute bars using a trend-following strategy, runs a web dashboard for monitoring, and includes a stock screener that picks the best ticker to trade each morning.

Built for IBKR paper trading. There is a safety check that aborts if it detects a live account.

## What it does

- Connects to TWS on localhost:7497
- Subscribes to 1-minute historical bars with live updates
- Computes indicators (EMA 9/21/50, RSI, MACD, VWAP, ATR, volume MA)
- Enters long or short positions when all entry conditions align
- Manages exits via stop loss, take profit, trailing stop, EMA crossover, and end-of-day close
- Runs a stock screener at 9:15 AM daily that picks the best ticker from a curated watchlist
- Serves a real-time web dashboard at localhost:8080 with candlestick chart, indicators, trade log, and signal panel
- Sits in the system tray as a background app

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server, WebSocket broadcaster, system tray icon, entry point |
| `bot_engine.py` | Core trading logic, IBKR connection, order execution, auto-scan |
| `screener.py` | Stock screener with scoring model, auto-scan function |
| `backtest.py` | Backtester that mirrors bot_engine.py strategy exactly |
| `backtest_tuning.py` | Extended backtester for parameter optimization |
| `bot.py` | Original standalone bot (kept as fallback, not used in production) |
| `static/index.html` | Dashboard UI — chart, indicators, trade log, signal panel |

## Strategy: V3 Trend Rider

### Entry conditions (all must be true)

**Long:**
- Price above EMA 50 (trend filter)
- EMA 9 above EMA 21 (momentum)
- Price above VWAP
- RSI between 35-60 (not overbought)
- MACD histogram crossing positive or building upward
- Volume above 20-bar average

**Short:** Mirror of long conditions (below EMA 50, EMA 9 below 21, below VWAP, RSI 40-65, MACD crossing negative).

### Exit logic

- **Stop loss:** 2x ATR below entry (long) or above entry (short)
- **Take profit:** 3:1 reward-to-risk ratio (6x ATR from entry)
- **Trailing stop:** When price moves 1.5x ATR in your favor, stop loss moves to breakeven
- **EMA crossover:** If EMAs cross against you while in profit, exit early
- **End of day:** All positions closed at 3:55 PM EST

### Risk management

- 2% of account risked per trade
- Max 4 trades per day
- 3-bar cooldown between trades
- Position size capped at 95% of account value
- Paper account verification on startup

### Order execution

Limit orders with a $0.02 buffer for price protection. If not filled within 5 seconds, the order is cancelled (no chasing).

## Stock Screener

The screener scores stocks on 5 metrics derived from correlation analysis against actual backtest P&L:

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| VWAP Crosses | 25% | How often price crosses VWAP per day. Fewer = more trending. Strongest predictor (r=-0.720) |
| Choppiness Index | 25% | Range-bound vs trending. Lower = better (r=-0.683) |
| Efficiency Ratio | 20% | Directional movement vs total movement. Higher = cleaner moves (r=+0.667) |
| ADX | 20% | Trend strength. Sweet spot is 25-35. Above 40 starts to hurt, above 50 is auto-disqualified |
| Follow-through | 10% | How often a move continues in the next bar (r=+0.520) |

**ADX hard cap at 50.** Any stock with ADX above 50 gets score 0 regardless of other metrics. Extreme ADX means violent whipsaws that the strategy can't handle.

**Minimum score: 60.** If no ticker scores above 60, Ravi sits out that day.

### Curated watchlist (11 tickers)

All backtested profitable with the V3 strategy:

```
TSLA, LNAI, SOUN, META, DASH, COIN, HOOD, RIVN, MU, AVGO, ROKU
```

The screener picks from this list daily. It does not scan the entire market.

## Development history and lessons learned

### V1 and V2

The original bot was a simple EMA crossover system. V2 added RSI and MACD filters. Both were long-only and had no real risk management beyond a fixed stop loss. They lost money consistently on choppy stocks.

### V3 Trend Rider

Added short selling, VWAP filter, EMA 50 trend filter, trailing stop, and the 3:1 take profit ratio. This was the first version that was consistently profitable on the right tickers.

### Trailing stop trigger bug

The most critical bug found during development. The backtest used `TRAILING_STOP_TRIGGER * atr` (1.5x ATR = 1.5 ATR from entry), but bot_engine.py used `sl_distance * TRAILING_STOP_TRIGGER` (2.0 ATR * 1.5 = 3.0 ATR from entry). This meant live trading would trigger trailing stops at twice the distance the backtest expected. Backtest results would have been meaningless. Fixed by aligning both to `TRAILING_STOP_TRIGGER * atr`.

### Optimization trap

Tried adding partial exits (sell half at 1.5R, hold rest with trailing), ATR quality filter, and a 30-minute opening skip. Each looked good in isolation but collectively they changed the strategy enough that backtest results no longer matched. Stripped all optimizations back to clean V3 and kept only the limit order buffer ($0.02), which is an execution improvement rather than a strategy change.

### Screener calibration

The screener was initially built on intuition (ADX, choppiness, etc.) and then calibrated against actual backtest P&L across 20 tickers. The correlation analysis revealed VWAP crosses as the strongest predictor, which was not obvious beforehand. Choppiness Index was second.

### ADX scoring problem

SMCI and RIOT scored well on the screener (high ADX = strong trend) but lost money. High ADX means the trend is strong, but if Ravi enters on the wrong side of a violent move, it gets stopped out fast. Tightened ADX scoring so that above 35 the score drops sharply, and above 40 it tanks. Later added a hard cap at ADX 50 after BMY (ADX 79) snuck through with a score of 60 despite extreme volatility.

### SOUN paradox

SOUN consistently scores low on the screener (~27) because its average metrics look choppy. But it made +$6,541 in backtests because it has intermittent clean trending windows within overall choppy behavior. Attempted to add a "best-day choppiness" metric to catch it, but this promoted SOFI from 58 to 63 (above the 60 threshold) and SOFI loses money. Reverted the change. Accepted that some winners will be missed — the screener's job is loss prevention, not prediction.

### 200-ticker stress test

Ran the screener against ~200 tickers across all sectors. Results:
- Stocks scoring below 45 lost an average of -$1,321 each (total -$121,563 across 92 tickers). The screener correctly avoids these.
- Stocks scoring 65+ went net negative (-$6,364) because FSLY scored 67.2 but lost -$6,581. One bad pick wiped out the entire tier.
- This confirmed the screener works as a filter on a curated watchlist, not as a universal stock picker. Expanding to random tickers dilutes the top tier with stocks that look good on metrics but still lose.

### Watchlist curation

Profiled the top 5 winners (TSLA, LNAI, SOUN, META, NVDA) by ATR%, daily volatility, average range, and beta. Scanned 70+ candidates for similar behavior profiles. Backtested the top 20 matches. Only 4 out of 20 were profitable (DASH, COIN, HOOD, RIVN). Added MU, AVGO, ROKU from a second batch. Final watchlist: 11 tickers, all backtested profitable.

### PyInstaller issues

- `from screener import auto_scan` inside a method (lazy import) was not detected by PyInstaller. Moved to top-level import and added `--hidden-import screener`.
- `--add-data "templates;templates"` failed because templates/ directory doesn't exist. Removed from build command.
- `sys.stdout` is None in windowed mode (`--noconsole`), which crashes uvicorn. Fixed by redirecting to `os.devnull`.

## Building the exe

```
python -m PyInstaller --onefile --noconsole --name ravioli --add-data "static;static" --hidden-import screener app.py
```

Output: `dist/ravioli.exe`

## Running

**From source:**
```
python app.py
```

**From exe:**
Double-click `ravioli.exe`. Dashboard opens at http://localhost:8080.

**Requirements:**
- TWS or IB Gateway running on localhost:7497
- Paper trading account
- Python 3.11+ with: ib_insync, fastapi, uvicorn, pandas, numpy, yfinance, pystray, Pillow

## Backtesting

```
python backtest.py TSLA 30        # Single ticker, 30 days
python backtest.py TSLA NVDA META  # Multiple tickers, default 7 days
```

## Screener

```
python screener.py                 # Scan default watchlist
python screener.py TSLA NVDA AAPL  # Scan specific tickers
```
