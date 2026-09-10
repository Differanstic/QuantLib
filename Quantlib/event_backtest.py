"""A small, stateful event-driven backtester for one net position at a time."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

from . import utils as u

LONG, SHORT = 1, -1
_SIDES = {"LONG": LONG, "SHORT": SHORT, LONG: LONG, SHORT: SHORT}


class BacktestValidationError(ValueError):
    """Raised when an instruction or market-data invariant is invalid."""


@dataclass
class BacktestContext:
    """Read-only, no-look-ahead strategy data access.

    ``column(name)`` returns a NumPy array and ``history(name)`` returns values
    through the current event only.  The full DataFrame is deliberately not
    exposed to new callbacks.
    """
    columns: Mapping[str, np.ndarray]
    index: int = 0

    def column(self, name: str) -> np.ndarray:
        return self.columns[name]

    def history(self, name: str, start: int = 0) -> np.ndarray:
        return self.columns[name][start:self.index + 1]

    def value(self, name: str) -> Any:
        return self.columns[name][self.index]


@dataclass
class Fill:
    price: float
    qty: float
    timestamp: Any
    idx: int
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    gross_pnl: Optional[float] = None
    charges: float = 0.0


@dataclass
class Trade:
    trade_id: int
    side: str
    direction: int
    entry_time: Any
    entry_idx: int
    initial_qty: float
    remaining_qty: float
    entries: list[Fill] = field(default_factory=list)
    exits: list[Fill] = field(default_factory=list)
    total_entry_qty: float = 0.0
    total_entry_notional: float = 0.0
    total_exit_qty: float = 0.0
    total_exit_notional: float = 0.0
    realized_pnl: float = 0.0
    charges: float = 0.0
    stop_price: Optional[float] = None
    tif: float = 0.0
    tia: float = 0.0
    mfe_points: float = 0.0
    mae_points: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    last_timestamp: Any = None
    triggered_exits: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    legacy_entry_row: Optional[dict[str, Any]] = None
    entry_record: dict[str, Any] = field(default_factory=dict)
    final_exit_record: dict[str, Any] = field(default_factory=dict)
    exit_reason: Optional[str] = None

    @property
    def avg_entry_price(self) -> float:
        return self.total_entry_notional / self.total_entry_qty

    @property
    def avg_exit_price(self) -> float:
        return self.total_exit_notional / self.total_exit_qty

    # Attribute retained for simple strategy code (trade.side, trade.entries etc.).
    @property
    def entry_price(self) -> float:
        return self.avg_entry_price


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    open_trade: Optional[Trade]


def _finite(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise BacktestValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise BacktestValidationError(f"{name} must be a finite number")
    return value


def _side(value: Any) -> tuple[str, int]:
    try:
        direction = _SIDES[str(value).upper()] if isinstance(value, str) else _SIDES[value]
    except KeyError as exc:
        raise BacktestValidationError("side must be LONG or SHORT") from exc
    return ("LONG" if direction == LONG else "SHORT"), direction


def _instructions(value: Any, kind: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(x, Mapping) for x in value):
        return list(value)
    raise BacktestValidationError(f"{kind} callback must return None, a mapping, or a list of mappings")


def _arity(fn: Callable[..., Any]) -> int:
    return len([p for p in inspect.signature(fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])


def _default_charges(entry_price: float, exit_price: float, qty: float) -> float:
    return float(u.calculate_options_charges(entry_price, exit_price, qty)["total_charges"])


def _append_entry(trade: Trade, instruction: Mapping[str, Any], price: float, timestamp: Any,
                  idx: int, record: dict[str, Any]) -> None:
    qty = _finite(instruction.get("qty"), "entry qty")
    if qty <= 0:
        raise BacktestValidationError("entry qty must be positive")
    fill_price = _finite(instruction.get("price", price), "entry price")
    fill = Fill(fill_price, qty, timestamp, idx,
                {k: v for k, v in instruction.items()
                 if k not in {"side", "qty", "price", "stop_price", "tag", "metadata"}})
    trade.entries.append(fill)
    trade.remaining_qty += qty
    trade.total_entry_qty += qty
    trade.total_entry_notional += fill_price * qty
    if instruction.get("tag") is not None:
        trade.metadata.setdefault("tags", []).append(instruction["tag"])
    if instruction.get("metadata") is not None:
        if not isinstance(instruction["metadata"], Mapping):
            raise BacktestValidationError("entry metadata must be a mapping")
        trade.metadata.update(instruction["metadata"])
    trade.entry_record = trade.entry_record or record


def _stop_valid(stop: Any, direction: int, price: float) -> float:
    stop = _finite(stop, "stop_price")
    if (direction == LONG and stop > price) or (direction == SHORT and stop < price):
        raise BacktestValidationError("stop_price must be at or beyond the adverse side of current price")
    return stop


def _close(trade: Trade, instruction: Mapping[str, Any], default_price: float, timestamp: Any,
           idx: int, record: dict[str, Any], charges_fn: Callable[[float, float, float], float]) -> bool:
    qty = _finite(instruction.get("qty", trade.remaining_qty), "exit qty")
    if qty <= 0:
        raise BacktestValidationError("exit qty must be positive")
    if qty > trade.remaining_qty + 1e-10:
        raise BacktestValidationError("exit qty cannot exceed remaining_qty")
    price = _finite(instruction.get("price", default_price), "exit price")
    cost = trade.avg_entry_price
    gross = (price - cost) * trade.direction * qty
    charge = _finite(charges_fn(cost, price, qty), "charges_fn result")
    reason = str(instruction.get("reason", "EXIT_FN"))
    fill = Fill(price, qty, timestamp, idx,
                {k: v for k, v in instruction.items() if k not in {"qty", "price", "reason"}},
                reason, gross, charge)
    trade.exits.append(fill)
    trade.remaining_qty = max(0.0, trade.remaining_qty - qty)
    trade.total_exit_qty += qty
    trade.total_exit_notional += price * qty
    trade.realized_pnl += gross
    trade.charges += charge
    trade.exit_reason = reason
    trade.final_exit_record = record
    if instruction.get("trigger"):
        trade.triggered_exits.add(str(instruction["trigger"]))
    return trade.remaining_qty <= 1e-10


def _row_record(record_arrays: Mapping[str, np.ndarray], i: int) -> dict[str, Any]:
    return {name: values[i] for name, values in record_arrays.items()}


def _trade_row(trade: Trade, record_cols: list[str]) -> dict[str, Any]:
    if abs(trade.total_exit_qty - trade.total_entry_qty) > 1e-8 or trade.remaining_qty > 1e-8:
        raise AssertionError("completed position has unreconciled quantity")
    gross_from_fills = sum(x.gross_pnl or 0.0 for x in trade.exits)
    if not math.isclose(gross_from_fills, trade.realized_pnl, abs_tol=1e-8):
        raise AssertionError("exit-fill PnL does not reconcile")
    exit_time, exit_idx = trade.exits[-1].timestamp, trade.exits[-1].idx
    entry_notional = trade.total_entry_notional
    result = {
        "trade_id": trade.trade_id, "side": trade.side, "direction": trade.direction,
        "entry_time": trade.entry_time, "exit_time": exit_time, "entry_idx": trade.entry_idx,
        "exit_idx": exit_idx, "initial_qty": trade.initial_qty, "remaining_qty": trade.remaining_qty,
        "closed_qty": trade.total_exit_qty, "total_entry_qty": trade.total_entry_qty,
        "total_exit_qty": trade.total_exit_qty, "entry_price": trade.avg_entry_price,
        "exit_price": trade.avg_exit_price, "avg_entry_price": trade.avg_entry_price,
        "avg_exit_price": trade.avg_exit_price, "gross_pnl": trade.realized_pnl,
        "charges": trade.charges, "pnl": trade.realized_pnl - trade.charges,
        "return_pct": (trade.realized_pnl / entry_notional * 100) if entry_notional else np.nan,
        "tif": trade.tif, "tia": trade.tia, "mfe_points": trade.mfe_points,
        "mae_points": trade.mae_points, "mfe_pct": trade.mfe_pct, "mae_pct": trade.mae_pct,
        "holding_time_sec": pd.Timedelta(exit_time - trade.entry_time).total_seconds(),
        "entry_count": len(trade.entries), "exit_count": len(trade.exits),
        "exit_reason": trade.exit_reason, "stop_price": trade.stop_price,
        "entries": [vars(x).copy() for x in trade.entries], "exits": [vars(x).copy() for x in trade.exits],
        "metadata": trade.metadata,
    }
    for col in record_cols:
        result[f"entry_{col}"] = trade.entry_record.get(col)
        result[f"exit_{col}"] = trade.final_exit_record.get(col)
    return result


def backtest(df: pd.DataFrame, entry_fn: Callable[..., Any], exit_fn: Callable[..., Any],
             record_col: Optional[list[str]] = None, tsl_fn: Optional[Callable[..., Any]] = None, *,
             adjust_fn: Optional[Callable[..., Any]] = None,
             charges_fn: Optional[Callable[[float, float, float], float]] = None,
             close_open_position: bool = True, allow_same_row_exit: bool = False,
             allow_stop_loosen: bool = False, return_open_position: bool = False,
             data_mode: str = "auto", price_col: str = "ltp") -> pd.DataFrame | BacktestResult:
    """Run a single-net-position event backtest.

    New callbacks use ``(row, i, context, state)`` for entries and
    ``(row, i, context, trade, state)`` for exit/adjust/TSL. ``state`` is a
    mutable dict intentionally shared for strategy flags.  ``exit_fn`` may
    return one exit mapping or a list.  ``adjust_fn`` returns scale-in entries.
    Legacy 3/5 argument entry/exit callbacks are accepted as a migration aid.

    ``data_mode`` is ``"tick"``, ``"ohlc"``, or ``"auto"`` (default).
    Tick data uses ``price_col`` (default ``ltp``). OHLC data requires lower-case
    ``open``, ``high``, ``low``, and ``close`` columns; callbacks execute using
    the close price and stop detection uses low for LONG / high for SHORT.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        empty = pd.DataFrame()
        return BacktestResult(empty, None) if return_open_position else empty
    mode = str(data_mode).lower()
    if mode not in {"auto", "tick", "ohlc"}:
        raise BacktestValidationError("data_mode must be 'auto', 'tick', or 'ohlc'")
    ohlc_required = {"open", "high", "low", "close"}
    if mode == "auto":
        mode = "tick" if price_col in df.columns else "ohlc" if ohlc_required.issubset(df.columns) else ""
    required = {"timestamp", price_col} if mode == "tick" else {"timestamp", *ohlc_required}
    missing = required.difference(df.columns)
    if missing:
        expected = f"tick data with '{price_col}'" if mode == "tick" else "OHLC data with open/high/low/close"
        raise BacktestValidationError(f"df is missing required columns for {expected}: {sorted(missing)}")
    record_col = list(record_col or [])
    unknown = set(record_col).difference(df.columns)
    if unknown:
        raise BacktestValidationError(f"record_col not in df: {sorted(unknown)}")
    work = df.reset_index(drop=True)
    price_source = price_col if mode == "tick" else "close"
    prices = pd.to_numeric(work[price_source], errors="coerce").to_numpy(dtype=float)
    highs = prices if mode == "tick" else pd.to_numeric(work["high"], errors="coerce").to_numpy(dtype=float)
    lows = prices if mode == "tick" else pd.to_numeric(work["low"], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_datetime(work["timestamp"], errors="coerce").to_numpy()
    if not np.isfinite(prices).all() or not np.isfinite(highs).all() or not np.isfinite(lows).all() or pd.isna(timestamps).any():
        raise BacktestValidationError("price/OHLC values and timestamp must not contain NaN/NaT")
    if mode == "ohlc":
        opens = pd.to_numeric(work["open"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(opens).all() or (highs < lows).any() or (highs < np.maximum(opens, prices)).any() or (lows > np.minimum(opens, prices)).any():
            raise BacktestValidationError("OHLC candles must satisfy low <= open/close <= high")
    if len(timestamps) > 1 and (timestamps[1:] < timestamps[:-1]).any():
        raise BacktestValidationError("timestamp cannot go backwards")
    columns = {name: work[name].to_numpy(copy=False) for name in work.columns}
    records = {name: columns[name] for name in record_col}
    context, state, charge = BacktestContext(columns), {}, charges_fn or _default_charges
    entry_arity = _arity(entry_fn)
    exit_arity = _arity(exit_fn)
    adjust_arity = _arity(adjust_fn) if adjust_fn else 0
    tsl_arity = _arity(tsl_fn) if tsl_fn else 0
    # A five-argument exit is ambiguous: it is the new API as well as the old
    # API.  Treat it as legacy only when paired with a legacy three-argument
    # entry callback.
    legacy = entry_arity <= 3
    rows = work.itertuples(index=False, name="Row")
    completed: list[dict[str, Any]] = []
    trade: Optional[Trade] = None
    next_id = 1

    for i, row in enumerate(rows):
        context.index = i
        price, timestamp, record = prices[i], timestamps[i], _row_record(records, i)
        opened_now = False
        if trade is None:
            raw = entry_fn(row, i, work) if entry_arity <= 3 else entry_fn(row, i, context, state)
            # Original ``(condition, lot_size)`` entries remain LONG-only.
            if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], (bool, np.bool_)):
                raw = {"side": "LONG", "qty": raw[1]} if raw[0] else None
            orders = _instructions(raw, "entry")
            if len(orders) > 1:
                raise BacktestValidationError("entry_fn may open one position; use adjust_fn for scale-ins")
            if orders:
                order = orders[0]
                side, direction = _side(order.get("side"))
                first_price = _finite(order.get("price", price), "entry price")
                first_qty = _finite(order.get("qty"), "entry qty")
                if first_qty <= 0:
                    raise BacktestValidationError("entry qty must be positive")
                trade = Trade(next_id, side, direction, timestamp, i, first_qty, 0.0, last_timestamp=timestamp)
                next_id += 1
                if legacy:
                    trade.legacy_entry_row = row._asdict()
                _append_entry(trade, order, price, timestamp, i, record)
                if order.get("stop_price") is not None:
                    trade.stop_price = _stop_valid(order["stop_price"], direction, first_price)
                opened_now = True
        if trade is None or (opened_now and not allow_same_row_exit):
            continue

        delta = pd.Timedelta(timestamp - trade.last_timestamp).total_seconds()
        if delta < 0:  # global validation makes this a defensive invariant.
            raise BacktestValidationError("timestamp cannot go backwards while a trade is open")
        trade.last_timestamp = timestamp
        signed_move = (price - trade.avg_entry_price) * trade.direction
        if signed_move > 0:
            trade.tif += delta
        elif signed_move < 0:
            trade.tia += delta
        favorable_price = highs[i] if trade.direction == LONG else lows[i]
        adverse_price = lows[i] if trade.direction == LONG else highs[i]
        favorable_move = (favorable_price - trade.avg_entry_price) * trade.direction
        adverse_move = (adverse_price - trade.avg_entry_price) * trade.direction
        trade.mfe_points = max(trade.mfe_points, favorable_move)
        trade.mae_points = min(trade.mae_points, adverse_move)
        base = trade.avg_entry_price
        trade.mfe_pct = max(trade.mfe_pct, favorable_move / base * 100)
        trade.mae_pct = min(trade.mae_pct, adverse_move / base * 100)

        if adjust_fn is not None:
            raw = adjust_fn(row, i, context, trade, state)
            for order in _instructions(raw, "adjust"):
                side, direction = _side(order.get("side", trade.side))
                if direction != trade.direction:
                    raise BacktestValidationError("cannot scale in with the opposite direction")
                _append_entry(trade, order, price, timestamp, i, record)

        if tsl_fn is not None:
            new_stop = tsl_fn(row, i, work, trade) if tsl_arity <= 4 else tsl_fn(row, i, context, trade, state)
            if new_stop is not None:
                new_stop = _stop_valid(new_stop, trade.direction, price)
                old = trade.stop_price
                worse = old is not None and ((trade.direction == LONG and new_stop < old) or
                                             (trade.direction == SHORT and new_stop > old))
                if worse and not allow_stop_loosen:
                    raise BacktestValidationError("trailing stop may only move favorably")
                trade.stop_price = new_stop

        stop_probe = lows[i] if trade.direction == LONG else highs[i]
        stopped = trade.stop_price is not None and ((trade.direction == LONG and stop_probe <= trade.stop_price) or
                                                     (trade.direction == SHORT and stop_probe >= trade.stop_price))
        if stopped:
            _close(trade, {"qty": trade.remaining_qty, "price": trade.stop_price, "reason": "TSL"}, price,
                   timestamp, i, record, charge)
            completed.append(_trade_row(trade, record_col))
            trade = None
            continue

        raw = exit_fn(row, i, work, trade.legacy_entry_row, trade.entry_idx) if legacy else exit_fn(row, i, context, trade, state)
        # Original ``(signal, price)`` exits close the whole remaining position.
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], (bool, np.bool_)):
            raw = {"qty": trade.remaining_qty, "price": raw[1], "reason": "EXIT_FN"} if raw[0] else None
        for order in _instructions(raw, "exit"):
            if _close(trade, order, price, timestamp, i, record, charge):
                completed.append(_trade_row(trade, record_col))
                trade = None
                break

    if trade is not None and close_open_position:
        _close(trade, {"qty": trade.remaining_qty, "price": prices[-1], "reason": "END_OF_DATA"}, prices[-1],
               timestamps[-1], len(work) - 1, _row_record(records, len(work) - 1), charge)
        completed.append(_trade_row(trade, record_col))
        trade = None
    results = pd.DataFrame(completed)
    return BacktestResult(results, trade) if return_open_position else results

