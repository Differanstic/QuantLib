import pandas as pd
import plotly.graph_objects as go
from . import utils as u

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


def backtest(df, entry_fn, exit_fn, lot_size=75, tsl_fn=None):
    """
    Generalized backtest with row-by-row entry/exit functions + optional trailing stoploss.
    
    Args:
        df (pd.DataFrame): must contain 'ltp'
        entry_fn (function): (row, i, df) -> bool (enter trade?)
        exit_fn (function): (row, i, df, entry_row, entry_idx) -> (bool, exit_price)
        lot_size (int): lot size for PnL calculation
        tsl_fn (function): (row, i, df, open_trade) -> new_stop_price or None
    
    Returns:
        trades (DataFrame), total_pnl (float)
    """
    df = df.reset_index(drop=True)
    
    trades = []
    total_pnl = 0
    open_trade = None
    
    for i, row in df.iterrows():
        price = row['ltp']
        
        # --- Entry ---
        if open_trade is None:
            if entry_fn(row, i, df):
                open_trade = {
                    "entry_idx": i,
                    "entry_price": price,
                    "entry_row": row.to_dict(),
                    "entry_time": row['timestamp'],
                    "stop_price": None  # for trailing SL
                }
            continue
        
        # --- Update Trailing Stop ---
        if tsl_fn is not None:
            new_stop = tsl_fn(row, i, df, open_trade)
            if new_stop is not None:
                open_trade["stop_price"] = new_stop
        
        # --- Check stoploss exit ---
        if open_trade["stop_price"] is not None and price <= open_trade["stop_price"]:
            exit_price = open_trade["stop_price"]
            charges = u.calculate_options_charges(open_trade["entry_price"], exit_price, lot_size)['total_charges']
            pnl = (((exit_price - open_trade["entry_price"]) * lot_size) - charges)
            
            open_trade.update({
                "exit_idx": i,
                "exit_price": exit_price,
                "exit_time": row['timestamp'],
                "pnl": pnl,
                "charges": charges,
                "exit_reason": "TSL"
            })
            trades.append(open_trade)
            total_pnl += pnl - charges
            open_trade = None
            continue
        
        # --- Normal Exit ---
        exit_signal, exit_price = exit_fn(row, i, df, open_trade["entry_row"], open_trade["entry_idx"])
        if exit_signal:
            charges = u.calculate_options_charges(open_trade["entry_price"], exit_price, lot_size)['total_charges']
            pnl = (((exit_price - open_trade["entry_price"]) * lot_size) - charges)
            open_trade.update({
                "exit_idx": i,
                "exit_price": exit_price,
                "exit_time": row['timestamp'],
                "pnl": pnl,
                "charges": charges,
                "exit_reason": "EXIT_FN"
            })
            trades.append(open_trade)
            total_pnl += pnl 
            open_trade = None
    
    return pd.DataFrame(trades), total_pnl

import plotly.express as px
def plot_trades(df, trades_df, additional_rows=[], showInBrowser=False):
    """
    Plot LTP with entry/exit markers from backtest trades using timestamp axis.
    
    Args:
        df (pd.DataFrame): must contain ['timestamp','ltp']
        trades_df (pd.DataFrame): output from backtest() with entry/exit idx
    """
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Map indices to timestamps
    #trades_df['entry_time'] = df.loc[trades_df['entry_idx'], 'timestamp'].values
    #trades_df['exit_time']  = df.loc[trades_df['exit_idx'], 'timestamp'].values

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['ltp'],
        mode='lines',
        name='LTP',
        line=dict(color='white')
    ))

    # Additional rows
    for row in additional_rows:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[row],
            mode='lines',
            name=row,
        ))

    # Entry points
    fig.add_trace(go.Scatter(
        x=trades_df['entry_time'],
        y=trades_df['entry_price'],
        mode='markers',
        name='Entry',
        marker=dict(symbol='triangle-up', size=12, color='green')
    ))

    # Exit points
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=trades_df['exit_price'],
        mode='markers+text',
        name='Exit',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        text=[f"PnL: {p:.0f}" for p in trades_df['pnl']],  # label with PnL
        textposition="top center"
    ))

    # Stats
    trades_df['net'] = trades_df['pnl'] - trades_df['charges']
    total_pnl = trades_df['pnl'].sum() - trades_df['charges'].sum()
    profit_prob = len(trades_df[trades_df['pnl']>0]) / len(trades_df) if len(trades_df) > 0 else 0
    loss_prob = len(trades_df[trades_df['pnl']<0]) / len(trades_df) if len(trades_df) > 0 else 0
    fig.add_annotation(
        text=f"Total PnL: {total_pnl:.2f},\n Profit: {trades_df[trades_df['pnl']>0]['pnl'].sum():.2f}({profit_prob:.2f}),\n Loss: {trades_df[trades_df['pnl']<0]['pnl'].sum():.2f}({loss_prob:.2f}),\n Charges: {trades_df['charges'].sum():.2f}",
        xref="paper", yref="paper",
        x=0.01, y=0.99, showarrow=False,
        font=dict(size=14, color="yellow"),
        bgcolor="black", bordercolor="yellow", borderwidth=1
    )

    # Layout
    fig.update_layout(
        title="Backtest Trades",
        xaxis_title="Timestamp",
        yaxis_title="LTP",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if showInBrowser:
        fig.show(renderer="browser")
    else:
        fig.show()
    trades_df['pnl_color'] = trades_df['pnl'].apply(lambda x: 'Profit' if x >= 0 else 'Loss')

    fig = px.histogram(
        trades_df,
        x='pnl',
        nbins=100,
        color='pnl_color',
        color_discrete_map={'Profit': 'green', 'Loss': 'red'},
        title='PnL Distribution'
    )

    fig.update_layout(showlegend=False)
    fig.show()
    return fig
