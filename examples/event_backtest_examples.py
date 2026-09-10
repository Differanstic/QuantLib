"""Runnable callback examples for Quantlib.backtest.backtest."""
import pandas as pd
from Quantlib.backtest import backtest


def long_entry(row, i, ctx, state):
    return {"side": "LONG", "qty": 75, "stop_price": row.ltp - 10, "tag": "BREAKOUT"} if row.pred_high - row.ltp > 20 else None


def short_entry(row, i, ctx, state):
    return {"side": "SHORT", "qty": 75, "stop_price": row.ltp + 10} if row.ltp - row.pred_low > 20 else None


def both_sides_entry(row, i, ctx, state):
    return long_entry(row, i, ctx, state) or short_entry(row, i, ctx, state)


def laddered_exit(row, i, ctx, trade, state):
    # ``trigger`` is atomically recorded by the engine after the fill.
    targets = (("TP1", 10, 25), ("TP2", 20, 25), ("TP3", 30, 25))
    for name, points, qty in targets:
        if name not in trade.triggered_exits and (row.ltp - trade.avg_entry_price) * trade.direction >= points:
            return {"qty": min(qty, trade.remaining_qty), "price": row.ltp, "reason": name, "trigger": name}
    return None


def scale_in(row, i, ctx, trade, state):
    # A scale-in is always in the same direction as the open position.
    if trade.side == "LONG" and row.ltp <= trade.avg_entry_price - 5 and "ADD1" not in trade.metadata:
        trade.metadata["ADD1"] = True
        return {"qty": 25, "tag": "ADD1"}
    return None


def trailing_stop(row, i, ctx, trade, state):
    # The engine rejects a worsening stop by default.
    if trade.side == "LONG":
        return row.ltp - 8
    return row.ltp + 8


def combined_entry(row, i, ctx, state):
    return {"side": "LONG", "qty": 50, "stop_price": row.ltp - 10} if i == 0 else None


if __name__ == "__main__":
    data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8, freq="min"),
        "ltp": [100, 95, 105, 111, 121, 115, 130, 120],
        "pred_high": [130] * 8,
        "pred_low": [70] * 8,
    })
    # Scale-in + laddered exits + trailing stop. End-of-data closes any remainder.
    print(backtest(data, combined_entry, laddered_exit, adjust_fn=scale_in,
                   tsl_fn=trailing_stop, close_open_position=True))
