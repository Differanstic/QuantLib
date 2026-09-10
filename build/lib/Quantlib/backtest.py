import pandas as pd
import plotly.graph_objects as go
from . import utils as u
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# Open-To-Close Strat
def intraday_open_to_close(df, lot_size=1):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    trades = {}
    entry_price = df['ltp'].iloc[0]
    exit_price  = df['ltp'].iloc[-1]
    pnl = (exit_price - entry_price) * lot_size
    ret = (exit_price / entry_price - 1) * 100
    trades ={
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "return_%": ret
    }
    return trades


def backteset(df, entry_fn, exit_fn, record_col=[], tsl_fn=None):
    """
    Universal backtest engine with:
    - entry_fn(row, i, df)
    - exit_fn(row, i, df, entry_row, entry_idx)
    - tsl_fn(row, i, df, trade_dict) for dynamic stop updates
    - automatic TIF / TIA tracking in seconds
    - record_col: list of df columns to store on entry & exit
    """

    

    if record_col is None:
        record_col = []

    df = df.reset_index(drop=True)

    trades = []
    total_pnl = 0
    open_trade = None       
    lot_size: int

    for i, row in df.iterrows():
        price = row["ltp"]
        timestamp = row["timestamp"]

        if open_trade is None:
            cond, lot_size = entry_fn(row, i, df)

            if cond:
                entry_values = {col: row[col] for col in record_col}

                open_trade = {
                    "entry_idx": i,
                    "entry_price": price,
                    "entry_row": row.to_dict(),
                    "entry_time": timestamp,
                    "stop_price": None,
                    "entry_values": entry_values,
                    'lot_size':lot_size,
                    # NEW → Time in favour/against in seconds
                    "tif": 0.0,
                    "tia": 0.0,
                    "last_timestamp": timestamp,
                }
            continue

       
        delta_sec = (timestamp - open_trade["last_timestamp"]).total_seconds()
        if delta_sec < 0:
            delta_sec = 0  # safety

        open_trade["last_timestamp"] = timestamp

       
        if price > open_trade["entry_price"]:
            open_trade["tif"] += delta_sec
        elif price < open_trade["entry_price"]:
            open_trade["tia"] += delta_sec
        

       
        if tsl_fn is not None:
            new_stop = tsl_fn(row, i, df, open_trade)
            if new_stop is not None:
                open_trade["stop_price"] = new_stop

        
        if open_trade["stop_price"] is not None and price <= open_trade["stop_price"]:
            exit_price = open_trade["stop_price"]

            charges = u.calculate_options_charges(
                open_trade["entry_price"], exit_price, lot_size
            )["total_charges"]

            pnl = (exit_price - open_trade["entry_price"]) * lot_size

            exit_values = {col: row[col] for col in record_col}

            open_trade.update({
                "exit_idx": i,
                "exit_price": exit_price,
                "exit_time": timestamp,
                "gross_pnl": pnl,
                "charges": charges,
                "exit_reason": "TSL",
                "exit_values": exit_values,
            })

            trades.append(open_trade)
            total_pnl += pnl 
            open_trade = None
            continue

        # ---------------------------------------------------------
        # NORMAL EXIT LOGIC USING exit_fn
        # ---------------------------------------------------------
        exit_signal, exit_price = exit_fn(
            row, i, df, open_trade["entry_row"], open_trade["entry_idx"]
        )

        if exit_signal:
            charges = u.calculate_options_charges(
                open_trade["entry_price"], exit_price, lot_size
            )["total_charges"]

            pnl = (exit_price - open_trade["entry_price"]) * lot_size

            exit_values = {col: row[col] for col in record_col}

            open_trade.update({
                "exit_idx": i,
                "exit_price": exit_price,
                "exit_time": timestamp,
                "gross_pnl": pnl,
                "charges": charges,
                "exit_reason": "EXIT_FN",
                "exit_values": exit_values,
            })

            trades.append(open_trade)
            total_pnl += pnl
            open_trade = None

    # ---------------------------------------------------------
    # CONVERT RESULTS TO DATAFRAME AND FLATTEN
    # ---------------------------------------------------------
    trades_df = pd.DataFrame(trades)

    if not len(trades_df) > 0:
        net_pnl = 0
    else:
        trades_df["pnl"] = (trades_df["gross_pnl"] - trades_df["charges"]).round(2)
        net_pnl = round(trades_df['pnl'].sum(),2)
        # Flatten recorded entry/exit columns
        for col in record_col:
            trades_df[f"entry_{col}"] = trades_df["entry_values"].apply(lambda d: d[col])
            trades_df[f"exit_{col}"]  = trades_df["exit_values"].apply(lambda d: d[col])
        trades_df = trades_df.drop(columns=["entry_values", "exit_values"])

    return trades_df





# Public event-driven engine. ``backteset`` above is the retained legacy
# single-long helper; use ``backtest`` below for the current API.
from .event_backtest import (
    LONG, SHORT, BacktestContext, BacktestResult, BacktestValidationError,
    Fill, Trade, backtest,
)


def trade_analysis(trades_df):
    """
    Calculate trade performance metrics from backtest output.
    """

    if trades_df is None or len(trades_df) == 0:
        return {
            "total_trades": 0,
            "net_pnl": 0,
        }

    df = trades_df.copy()

    # ---------------------------------------------------------
    # BASIC
    # ---------------------------------------------------------
    total_trades = len(df)

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    breakeven = df[df["pnl"] == 0]

    win_count = len(wins)
    loss_count = len(losses)
    breakeven_count = len(breakeven)

    win_rate = win_count / total_trades
    loss_rate = loss_count / total_trades

    # ---------------------------------------------------------
    # PNL
    # ---------------------------------------------------------
    gross_profit = wins["pnl"].sum()
    gross_loss = losses["pnl"].sum()

    net_pnl = df["pnl"].sum()

    avg_pnl = df["pnl"].mean()

    avg_win = wins["pnl"].mean() if win_count else 0
    avg_loss = losses["pnl"].mean() if loss_count else 0

    largest_win = df["pnl"].max()
    largest_loss = df["pnl"].min()

    # ---------------------------------------------------------
    # PROFIT FACTOR
    # ---------------------------------------------------------
    if gross_loss != 0:
        profit_factor = gross_profit / abs(gross_loss)
    else:
        profit_factor = np.inf

    # ---------------------------------------------------------
    # PAYOFF RATIO
    # ---------------------------------------------------------
    if avg_loss != 0:
        payoff_ratio = avg_win / abs(avg_loss)
    else:
        payoff_ratio = np.inf

    # ---------------------------------------------------------
    # EXPECTANCY
    #
    # E = P(win)*AvgWin + P(loss)*AvgLoss
    # ---------------------------------------------------------
    expectancy = (
        win_rate * avg_win +
        loss_rate * avg_loss
    )

    # ---------------------------------------------------------
    # EQUITY CURVE
    # ---------------------------------------------------------
    equity = df["pnl"].cumsum()

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_drawdown = abs(drawdown.min())

    # Drawdown %
    if running_max.max() != 0:
        drawdown_pct = drawdown / running_max.abs().replace(0, np.nan)
        max_drawdown_pct = abs(drawdown_pct.min()) * 100
    else:
        max_drawdown_pct = 0

    # ---------------------------------------------------------
    # RECOVERY FACTOR
    # ---------------------------------------------------------
    if max_drawdown != 0:
        recovery_factor = net_pnl / max_drawdown
    else:
        recovery_factor = np.inf

    # ---------------------------------------------------------
    # SHARPE RATIO
    #
    # Trade-level Sharpe
    # ---------------------------------------------------------
    pnl_std = df["pnl"].std()

    if pnl_std != 0 and not np.isnan(pnl_std):
        sharpe = df["pnl"].mean() / pnl_std * np.sqrt(total_trades)
    else:
        sharpe = 0

    # ---------------------------------------------------------
    # SORTINO RATIO
    # ---------------------------------------------------------
    downside = df.loc[df["pnl"] < 0, "pnl"]

    downside_std = downside.std()

    if downside_std != 0 and not np.isnan(downside_std):
        sortino = df["pnl"].mean() / downside_std * np.sqrt(total_trades)
    else:
        sortino = 0

    # ---------------------------------------------------------
    # CONSECUTIVE WINS / LOSSES
    # ---------------------------------------------------------
    result = np.sign(df["pnl"])

    max_consecutive_wins = 0
    max_consecutive_losses = 0

    current_wins = 0
    current_losses = 0

    for r in result:

        if r > 0:
            current_wins += 1
            current_losses = 0

            max_consecutive_wins = max(
                max_consecutive_wins,
                current_wins
            )

        elif r < 0:
            current_losses += 1
            current_wins = 0

            max_consecutive_losses = max(
                max_consecutive_losses,
                current_losses
            )

        else:
            current_wins = 0
            current_losses = 0

    # ---------------------------------------------------------
    # HOLDING TIME
    # ---------------------------------------------------------
    if "entry_time" in df.columns and "exit_time" in df.columns:

        holding_time = (
            pd.to_datetime(df["exit_time"]) -
            pd.to_datetime(df["entry_time"])
        ).dt.total_seconds()

        avg_holding_time = holding_time.mean()
        median_holding_time = holding_time.median()

    else:
        avg_holding_time = np.nan
        median_holding_time = np.nan

    # ---------------------------------------------------------
    # TIF / TIA
    # ---------------------------------------------------------
    avg_tif = df["tif"].mean() if "tif" in df else np.nan
    avg_tia = df["tia"].mean() if "tia" in df else np.nan

    win_avg_tif = (
        wins["tif"].mean()
        if "tif" in wins and len(wins)
        else np.nan
    )

    loss_avg_tia = (
        losses["tia"].mean()
        if "tia" in losses and len(losses)
        else np.nan
    )

    # ---------------------------------------------------------
    # CHARGES
    # ---------------------------------------------------------
    total_charges = (
        df["charges"].sum()
        if "charges" in df
        else 0
    )

    gross_before_charges = (
        df["gross_pnl"].sum()
        if "gross_pnl" in df
        else net_pnl + total_charges
    )

    # ---------------------------------------------------------
    # RETURN
    # ---------------------------------------------------------

    return {

        # Trades
        "total_trades": total_trades,
        "winning_trades": win_count,
        "losing_trades": loss_count,
        "breakeven_trades": breakeven_count,

        # Hit rate
        "win_rate": round(win_rate * 100, 2),
        "loss_rate": round(loss_rate * 100, 2),

        # PNL
        "gross_pnl": round(gross_before_charges, 2),
        "net_pnl": round(net_pnl, 2),
        "total_charges": round(total_charges, 2),

        "avg_trade": round(avg_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),

        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),

        # Quality
        "profit_factor": round(profit_factor, 3),
        "payoff_ratio": round(payoff_ratio, 3),
        "expectancy": round(expectancy, 2),

        # Risk
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),

        "recovery_factor": round(recovery_factor, 3),

        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),

        # Streaks
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,

        # Time
        "avg_holding_time_sec": round(avg_holding_time, 2),
        "median_holding_time_sec": round(median_holding_time, 2),

        # TIF / TIA
        "avg_tif_sec": round(avg_tif, 2),
        "avg_tia_sec": round(avg_tia, 2),

        "winner_avg_tif_sec": round(win_avg_tif, 2),
        "loser_avg_tia_sec": round(loss_avg_tia, 2),
    }



def plot_trades(
    df,
    trades_df,
    overlay_columns=[],   # plotted on main chart
    subplot_columns=[],   # plotted on separate subplot
    showInBrowser=False
):
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Prepare subplots (2 rows)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]  # adjust ratio
    )

    # ===========================
    # ROW 1 → PRICE CHART
    #===========================

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ltp'],
            mode='lines',
            name='LTP',
            line=dict(color='white')
        ),
        row=1, col=1
    )

    # Overlay indicators on MAIN chart
    for col in overlay_columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df[col],
                mode='lines',
                name=col,
            ),
            row=1, col=1
        )

    # Entry markers
    fig.add_trace(
        go.Scatter(
            x=trades_df['entry_time'],
            y=trades_df['entry_price'],
            mode='markers',
            name='Entry',
            marker=dict(symbol='triangle-up', size=12, color='green')
        ),
        row=1, col=1
    )

    # Exit markers
    fig.add_trace(
        go.Scatter(
            x=trades_df['exit_time'],
            y=trades_df['exit_price'],
            mode='markers+text',
            name='Exit',
            marker=dict(symbol='triangle-down', size=12, color='red'),
            text=[f"{p:.0f}" for p in trades_df['pnl']],
            textposition="top center"
        ),
        row=1, col=1
    )

    # ===========================
    # ROW 2 → SUBGRAPH (Different magnitude)
    #===========================

    if len(subplot_columns) > 0:
        for col in subplot_columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[col],
                    mode="lines",
                    name=f"{col} (subplot)"
                ),
                row=2, col=1
            )

    # ===========================
    # STAT BOX
    #===========================

    
    total_pnl = trades_df['pnl'].sum()
    profit_prob = (trades_df['pnl'] > 0).mean()
    loss_prob = (trades_df['pnl'] < 0).mean()

    fig.add_annotation(
        text=(
            f"Total PnL: {total_pnl:.2f}, "
            f"Profit: {trades_df[trades_df['gross_pnl']>0]['gross_pnl'].sum():.2f} ({profit_prob:.2f}), "
            f"Loss: {trades_df[trades_df['gross_pnl']<0]['gross_pnl'].sum():.2f} ({loss_prob:.2f}), "
            f"Charges: {trades_df['charges'].sum():.2f}"
        ),
        xref="paper", yref="paper",
        x=0.01, y=0.98, showarrow=False,
        font=dict(size=14, color="yellow"),
        bgcolor="black",
        bordercolor="yellow",
        borderwidth=1,
    )

    # ===========================
    # LAYOUT
    #===========================

    fig.update_layout(
        title="Backtest Trades (with Subplot Indicators)",
        template="plotly_dark",
        xaxis_title="Timestamp",
        legend=dict(orientation="h", yanchor="bottom", y=1.1)
    )

    # Display chart
    if showInBrowser:
        fig.show(renderer="browser")
    else:
        fig.show()

    return fig
