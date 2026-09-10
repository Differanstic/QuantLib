# Quantlib
---
Developed For Quantitative Analysis For Stock Market

## Event-driven backtesting

`Quantlib.backtest.backtest` is a lightweight stateful backtester for a single
net position. It supports LONG and SHORT positions, scale-ins, partial exits,
trailing stops, charges, tick data, OHLC candles, and detailed fill history. It
returns one row for each completed position (not one row per exit fill).

```python
import pandas as pd
from Quantlib.backtest import backtest

prices = pd.DataFrame({
    "timestamp": pd.date_range("2025-01-01 09:15", periods=6, freq="min"),
    "ltp": [100, 96, 106, 112, 122, 115],
    "pred_high": [130] * 6,
})

def entry_fn(row, i, ctx, state):
    if row.pred_high - row.ltp > 20:
        return {
            "side": "LONG",       # "LONG" or "SHORT"
            "qty": 50,
            "stop_price": row.ltp - 10,
            "tag": "BREAKOUT",
            "metadata": {"model": "v1"},
        }
    return None

def adjust_fn(row, i, ctx, trade, state):
    # Add to the existing position. Its side must match trade.side.
    if row.ltp <= trade.avg_entry_price - 4 and "added" not in trade.metadata:
        trade.metadata["added"] = True
        return {"qty": 25, "tag": "DIP_ADD"}
    return None

def exit_fn(row, i, ctx, trade, state):
    # ``trigger`` prevents the same target from filling again on later rows.
    if "TP1" not in trade.triggered_exits and row.ltp >= trade.avg_entry_price + 10:
        return {"qty": 25, "price": row.ltp, "reason": "TP1", "trigger": "TP1"}
    if "TP2" not in trade.triggered_exits and row.ltp >= trade.avg_entry_price + 20:
        return {"qty": 25, "price": row.ltp, "reason": "TP2", "trigger": "TP2"}
    return None

def tsl_fn(row, i, ctx, trade, state):
    # LONG stops can only move up; SHORT stops can only move down by default.
    return row.ltp - 8 if trade.side == "LONG" else row.ltp + 8

trades = backtest(
    prices,
    entry_fn,
    exit_fn,
    adjust_fn=adjust_fn,
    tsl_fn=tsl_fn,
    record_col=["pred_high"],
    close_open_position=True,
)
print(trades[["side", "avg_entry_price", "avg_exit_price", "gross_pnl", "charges", "pnl"]])
```

### Tick versus OHLC candle data

Pass the format explicitly with `data_mode`; `"auto"` (the default) detects
tick data when `ltp` exists and otherwise detects lower-case OHLC columns.

```python
# Tick data: timestamp + ltp (or use price_col for a differently named price column)
tick_trades = backtest(ticks, entry_fn, exit_fn, data_mode="tick", price_col="ltp")

# Candle data: timestamp + open, high, low, close
candle_trades = backtest(candles, entry_fn, exit_fn, data_mode="ohlc")
```

For OHLC candles, callbacks receive the candle row and use `close` as the
current execution price. A long stop is triggered when `low <= stop_price`; a
short stop is triggered when `high >= stop_price`, and both fill at the stop
price. MFE/MAE use candle high/low, while TIF/TIA use the close. Stops are
processed before normal callback exits, a conservative assumption because the
intrabar order of high and low is unknowable from OHLC data alone.

### Callback API

New callbacks receive a fast named tuple `row`, the integer row index `i`, and
a `BacktestContext` named `ctx`. `ctx.history("column")` exposes only data up
to the current row, helping prevent accidental look-ahead bias. `state` is a
dict shared across the run for strategy-level flags.

| Callback | Signature | Return value |
| --- | --- | --- |
| Entry | `entry_fn(row, i, ctx, state)` | `None` or `{side, qty, price?, stop_price?, tag?, metadata?}` |
| Exit | `exit_fn(row, i, ctx, trade, state)` | `None`, one `{qty, price?, reason?, trigger?}`, or a list of them |
| Scale-in | `adjust_fn(row, i, ctx, trade, state)` | `None`, one entry instruction, or a list of entry instructions |
| Trailing stop | `tsl_fn(row, i, ctx, trade, state)` | New stop price or `None` |

`price` defaults to `row.ltp`. It represents an immediate simulated fill; it
does not implement conditional limit-order matching. `qty` must be positive.
An exit cannot exceed `trade.remaining_qty`.

`trade` exposes `side`, `direction`, `avg_entry_price`, `remaining_qty`,
`entries`, `exits`, `triggered_exits`, and `metadata`. Use `triggered_exits` or
`state` to make laddered targets one-shot.

### Short example

```python
def short_entry(row, i, ctx, state):
    return {"side": "SHORT", "qty": 75, "stop_price": row.ltp + 10} \
        if row.ltp - row.pred_low > 20 else None

def short_exit(row, i, ctx, trade, state):
    if row.ltp <= trade.avg_entry_price - 15:
        return {"qty": trade.remaining_qty, "reason": "SHORT_TARGET"}
    return None
```

For shorts, profit is calculated as `(exit_price - entry_price) * -1 * qty`.
Stops trigger when `ltp >= stop_price`; for longs they trigger when
`ltp <= stop_price`.

### Results and configuration

The completed-trades DataFrame includes `trade_id`, `side`, times and indexes,
initial/entered/exited quantities, weighted average entry and exit prices,
`gross_pnl`, `charges`, `pnl`, return, TIF/TIA, MFE/MAE, fill counts,
`exit_reason`, `entries`, and `exits`. With `record_col=["atr"]`, it also adds
`entry_atr` and `exit_atr`; the exit value is from the final exit row.

- `charges_fn(entry_price, exit_price, qty)` overrides the default options
  charge calculation. It is called once for each closing fill.
- `close_open_position=True` (default) closes any remainder at the final LTP
  with `END_OF_DATA`. Set it to `False` and `return_open_position=True` to get
  `BacktestResult(trades, open_trade)`.
- `allow_same_row_exit=False` (default) disallows entry and exit on one row.
- `allow_stop_loosen=False` (default) enforces favorable-only trailing stops.

The old `entry_fn(row, i, df) -> (condition, qty)` and matching old five-argument
exit callback are supported as a migration aid, but the new context API is
recommended because it avoids accidental future-data access.

See `examples/event_backtest_examples.py` for long, short, combined,
laddered-exit, scale-in, trailing-stop, and combined strategy examples.
