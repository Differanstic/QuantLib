import pandas as pd
import plotly.graph_objects as go
from . import utils as u
import plotly.graph_objects as go
from plotly.subplots import make_subplots



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

import pandas as pd
def backtest(df, entry_fn, exit_fn, record_col=[], tsl_fn=None):
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

        # ---------------------------------------------------------
        # ENTRY LOGIC
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # CALCULATE TIME DELTA (SECONDS)
        # ---------------------------------------------------------
        delta_sec = (timestamp - open_trade["last_timestamp"]).total_seconds()
        if delta_sec < 0:
            delta_sec = 0  # safety

        open_trade["last_timestamp"] = timestamp

        # ---------------------------------------------------------
        # UPDATE TIF / TIA BASED ON CURRENT PRICE
        # ---------------------------------------------------------
        if price > open_trade["entry_price"]:
            open_trade["tif"] += delta_sec
        elif price < open_trade["entry_price"]:
            open_trade["tia"] += delta_sec
        

        # ---------------------------------------------------------
        # UPDATE STOPLOSS USING TSL FUNCTION
        # ---------------------------------------------------------
        if tsl_fn is not None:
            new_stop = tsl_fn(row, i, df, open_trade)
            if new_stop is not None:
                open_trade["stop_price"] = new_stop

        # ---------------------------------------------------------
        # STOPLOSS EXIT CHECK
        # ---------------------------------------------------------
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

    return trades_df, net_pnl



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
