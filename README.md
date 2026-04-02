# Ravioli Trading Bot

Automated day trading bot for Interactive Brokers — 1-minute bar trend-following strategy, built-in stock screener, real-time web dashboard, and packaged as a standalone `.exe`.

**Paper trading only.** Startup aborts if a live account is detected.

---

## Quick Start

**Requirements:**
- TWS or IB Gateway running on `localhost:7497`
- IBKR paper trading account
- Python 3.11+ with: `ib_insync fastapi uvicorn pandas numpy yfinance pystray Pillow`

**From source:**
```bash
python app.py
```

**From exe:** Double-click `ravioli.exe`. Dashboard opens at `http://localhost:8080`.

---

## Dashboard

Served at `http://localhost:8080`. Includes live candlestick chart with EMA/VWAP overlays, RSI/MACD/volume indicators, real-time trade log, and signal panel.

<img width="1919" height="945" alt="image" src="https://github.com/user-attachments/assets/6a75840d-60cc-4a1c-a688-dd7867928c3e" />

---

## Strategy: V3 Trend Rider

### Entry Conditions (all must be true)

**Long:**
- Price above EMA 50 (trend filter)
- EMA 9 above EMA 21 (momentum)
- Price above VWAP
- RSI between 35–60
- MACD histogram crossing positive or building upward
- Volume above 20-bar average

**Short:** Mirror of long conditions.

### Exit Logic

| Trigger | Detail |
|---|---|
| Stop loss | 2× ATR below/above entry |
| Take profit | 3:1 R ratio (6× ATR) |
| Trailing stop | Moves to breakeven when 1.5× ATR in your favor |
| EMA crossover | Exit early if EMAs cross against you while in profit |
| End of day | All positions closed at 3:55 PM EST |

### Risk Management

- 2% of account risked per trade
- Max 4 trades per day
- 3-bar cooldown between trades
- Position size capped at 95% of account
- Limit orders with $0.02 buffer; cancelled after 5 seconds if unfilled

---

## Stock Screener

Runs at 9:15 AM daily. Scores each ticker in the watchlist on 5 metrics derived from correlation analysis against actual backtest P&L:

| Metric | Weight | What It Measures |
|---|---|---|
| VWAP Crosses | 25% | Fewer = more trending (r = -0.720) |
| Choppiness Index | 25% | Lower = cleaner directional moves (r = -0.683) |
| Efficiency Ratio | 20% | Directional vs total movement — higher is better (r = +0.667) |
| ADX | 20% | Trend strength — sweet spot 25–35, hard cap at 50 |
| Follow-through | 10% | How often a move continues in the next bar (r = +0.520) |

Minimum score: **60**. If nothing clears the threshold, Ravioli sits out for the day.

**Watchlist (11 tickers — all backtested profitable with V3):**
`TSLA, LNAI, SOUN, META, DASH, COIN, HOOD, RIVN, MU, AVGO, ROKU`

---

## Project Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI server, WebSocket broadcaster, system tray icon, entry point |
| `bot_engine.py` | Core trading logic, IBKR connection, order execution, auto-scan |
| `screener.py` | Stock screener with scoring model |
| `backtest.py` | Backtester that mirrors `bot_engine.py` strategy exactly |
| `backtest_tuning.py` | Extended backtester for parameter optimization |
| `bot.py` | Original standalone bot (kept as fallback, not used in production) |
| `static/index.html` | Dashboard UI |

---

## Backtesting & Screener

```bash
python backtest.py TSLA 30         # Single ticker, 30 days
python backtest.py TSLA NVDA META  # Multiple tickers, default 7 days
python screener.py                 # Scan default watchlist
python screener.py TSLA NVDA AAPL  # Scan specific tickers
```

---

## Build

```bash
python -m PyInstaller --onefile --noconsole --name ravioli --add-data "static;static" --hidden-import screener app.py
```

Output: `dist/ravioli.exe`

---

## Development History & Lessons Learned

### V1 / V2
Simple EMA crossover system, then RSI/MACD filters added. Long-only, no real risk management. Lost money consistently on choppy stocks.

### V3 Trend Rider
Added short selling, VWAP filter, EMA 50 trend filter, trailing stop, and the 3:1 take profit ratio. First version that was consistently profitable on the right tickers.

---

### Key Bugs & Decisions

**Trailing stop trigger bug**
`backtest.py` used `TRAILING_STOP_TRIGGER × ATR` (1.5× ATR from entry). `bot_engine.py` used `sl_distance × TRAILING_STOP_TRIGGER` (2.0 ATR × 1.5 = **3.0× ATR**). Live trading would have triggered trailing stops at twice the backtest's expected distance, making all backtest results meaningless. Fixed by aligning both to `TRAILING_STOP_TRIGGER × ATR`.

**Optimization trap**
Tested partial exits, an ATR quality filter, and a 30-minute opening skip. Each looked good in isolation. Combined, they changed the strategy enough that backtests no longer reflected live behavior. Stripped everything back to clean V3. Only retained the $0.02 limit order buffer — an execution improvement, not a strategy change.

**Screener calibration**
Built initially on intuition, then validated against real backtest P&L across 20 tickers. Correlation analysis revealed VWAP Crosses as the strongest predictor — not obvious beforehand. ADX scoring was tightened after SMCI and RIOT scored well on raw ADX but still lost money (high ADX = strong trend, but entering on the wrong side of a violent move means fast stopouts).

**The SOUN paradox**
SOUN consistently scores ~27 (choppy on average metrics) but returned +$6,541 in backtests due to intermittent clean trending windows. Adding a "best-day choppiness" metric to capture it accidentally promoted SOFI past the 60 threshold — and SOFI loses money. Reverted. The screener's job is loss prevention, not prediction.

**200-ticker stress test**
Stocks scoring below 45 lost an average of -$1,321 each. Stocks scoring 65+ went net negative because FSLY (score 67.2) lost -$6,581 alone, wiping the entire tier. Conclusion: the screener works as a filter on a curated watchlist, not as a universal stock picker.

**Watchlist curation**
Profiled top winners by ATR%, daily volatility, average range, and beta. Scanned 70+ candidates. Backtested top 20 matches — only 4 were profitable. Final 11-ticker watchlist built from two batches, all with confirmed V3 profitability.

---

## Disclaimer

Personal learning project. Not financial advice. Not intended for use with real money.
