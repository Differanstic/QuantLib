import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import webbrowser


def plot_df(df):
    

    # Generate dark-styled table
    table_html = df[['timestamp','bp','ltp','sp','ltq']].style.set_table_attributes('border="1" class="dataframe table"') \
        .set_table_styles([
            {'selector': 'thead', 'props': [('background-color', '#1e1e1e'), ('color', 'white')]},
            {'selector': 'tbody', 'props': [('background-color', '#2e2e2e'), ('color', 'white')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#383838')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#505050')]}
        ]).to_html()

    # Wrap in full HTML with dark body
    html_content = f"""
    <html>
    <head>
        <title>Dark Table</title>
        <style>
            body {{
                background-color: #121212;
                color: white;
                font-family: Arial, sans-serif;
            }}
            h1 {{
                text-align: center;
                color: #ffffff;
            }}
            table {{
                width: 80%;
                margin: auto;
                border-collapse: collapse;
            }}
        </style>
    </head>
    <body>
        <h1>CE Option Data</h1>
        {table_html}
    </body>
    </html>
    """

    # Save to file
    with open("dark_table.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    # Open in browser
    webbrowser.open("dark_table.html")



def subplots(plot:list,showInBrowser:bool=False):
    """
    Creates and displays multiple subplots using Plotly, with each subplot containing one or more line plots.
    Args:
        plot (list): A list where each element is a tuple or list structured as (x_values, y_values_list, names_list, colors_list).
            - x_values (list): The x-axis values for the subplot.
            - y_values_list (list of lists): Each inner list contains y-axis values for a line in the subplot.
            - names_list (list): Names for each line in the subplot.
            - colors_list (list): Colors for each line in the subplot.
        showInBrowser (bool, optional): If True, displays the plot in the default web browser. If False, displays inline or in the default viewer. Defaults to False.
    Returns:
        None: The function displays the generated subplots and does not return any value.
    """
    
    fig = make_subplots(rows=len(plot), cols=1, shared_xaxes=True, vertical_spacing=0.01,subplot_titles=('Title'))
    fig.update_layout(template='plotly_dark')
    plot_number = 1
    for element in plot:
        for i in range(len(element[1])):
            fig.add_trace(go.Scatter(x=element[0], y=element[1][i], name=element[2][i], line=dict(color=element[3][i])), row=plot_number, col=1)
        plot_number += 1
    
    if showInBrowser:
        fig.show(renderer="browser")
    else:
        fig.show()
    
    return fig

def plot_candlestick(df,price_col='ltp',timeframe='15min',showinBrowser= False,plot_name='CandleStick'):
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    candles = df.set_index('timestamp').resample(timeframe)[price_col].ohlc().dropna()
    candles = candles.reset_index()

    # Plot candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=candles['timestamp'],
        open=candles['open'],
        high=candles['high'],
        low=candles['low'],
        close=candles['close'],
        name=plot_name
    )])

    fig.update_layout(
        title="Nifty Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    if showinBrowser:
        fig.show(renderer="browser")
    else:
        fig.show()