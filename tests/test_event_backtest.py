import pandas as pd

from Quantlib.event_backtest import BacktestValidationError, backtest


ZERO_CHARGES = lambda entry, exit, qty: 0.0


def test_scaled_weighted_average_and_laddered_accounting():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="min"),
        "ltp": [100, 90, 110, 120, 130],
        "feature": range(5),
    })
    def entry(row, i, ctx, state):
        return {"side": "LONG", "qty": 50} if i == 0 else None
    def adjust(row, i, ctx, trade, state):
        return {"qty": 50} if i == 1 else None
    def exit_fn(row, i, ctx, trade, state):
        if i == 2: return {"qty": 25, "price": 110, "reason": "TP1", "trigger": "TP1"}
        if i == 3: return {"qty": 25, "price": 120, "reason": "TP2", "trigger": "TP2"}
        return None
    result = backtest(df, entry, exit_fn, adjust_fn=adjust, charges_fn=ZERO_CHARGES, record_col=["feature"])
    trade = result.iloc[0]
    # (50*100 + 50*90) / 100 = 95; 25@110, 25@120, 50@130 => 2,750.
    assert trade.avg_entry_price == 95
    assert trade.total_entry_qty == trade.total_exit_qty == 100
    assert trade.remaining_qty == 0
    assert trade.gross_pnl == 2750
    assert sum(fill["gross_pnl"] for fill in trade.exits) == trade.gross_pnl
    assert trade.entry_feature == 0 and trade.exit_feature == 4


def test_short_stop_and_validation():
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=2, freq="min"), "ltp": [100, 106]})
    result = backtest(df, lambda r, i, c, s: {"side": "SHORT", "qty": 10, "stop_price": 105} if i == 0 else None,
                      lambda *args: None, charges_fn=ZERO_CHARGES)
    assert result.iloc[0].exit_reason == "TSL"
    assert result.iloc[0].gross_pnl == -50
    try:
        backtest(df, lambda r, i, c, s: {"side": "LONG", "qty": 0} if i == 0 else None,
                 lambda *args: None)
    except BacktestValidationError:
        pass
    else:
        raise AssertionError("zero quantity must be rejected")


def test_ohlc_stop_uses_low_and_mfe_uses_high():
    candles = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=2, freq="min"),
        "open": [100, 105], "high": [101, 120], "low": [99, 94], "close": [100, 110],
    })
    result = backtest(
        candles,
        lambda r, i, c, s: {"side": "LONG", "qty": 10, "stop_price": 95} if i == 0 else None,
        lambda *args: None,
        data_mode="ohlc", charges_fn=ZERO_CHARGES,
    )
    trade = result.iloc[0]
    assert trade.exit_reason == "TSL"
    assert trade.exit_price == 95  # candle low crossed the stop, not its close
    assert trade.mfe_points == 20  # 120 high versus 100 entry
    assert trade.mae_points == -6  # 94 low versus 100 entry
