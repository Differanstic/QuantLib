def calculate_options_charges(buy_price, sell_price, quantity, brokerage = 20, exchange_txn_charges = 0.0003503,sebi_charges_per = 0.000001,stamp_duty_per = 0.00003):
    
    """
    Calculate the detailed breakdown of charges and profit/loss for an options trade.
    Parameters:
        buy_price (float): The price at which the option was bought.
        sell_price (float): The price at which the option was sold.
        quantity (int): The number of option contracts traded.
        brokerage (float, optional): The fixed brokerage fee per trade. Default is 20.
        exchange_txn_charges (float, optional): Exchange transaction charges as a decimal percentage of turnover. Default is 0.0003503.
        sebi_charges_per (float, optional): SEBI charges as a decimal percentage of turnover. Default is 0.000001.
        stamp_duty_per (float, optional): Stamp duty as a decimal percentage of buy side turnover. Default is 0.00003.
    Returns:
        dict: A dictionary containing the following keys:
            - 'turnover': Total turnover for the trade.
            - 'brokerage': Brokerage charges applied.
            - 'stt': Securities Transaction Tax (STT) on the sell side.
            - 'exchange_txn_charges': Exchange transaction charges.
            - 'sebi_charges': SEBI charges.
            - 'gst': Goods and Services Tax (GST) on brokerage and exchange charges.
            - 'stamp_duty': Stamp duty on the buy side.
            - 'total_charges': Sum of all charges.
            - 'gross_profit': Gross profit before charges.
            - 'net_profit': Net profit after all charges.
            - 'points_to_breakeven': Points required to breakeven per contract.
    """

    
    
    turnover = (buy_price + sell_price) * quantity
    # STT on sell side only, 0.1% of sell premium
    stt = 0.001 * sell_price * quantity 
    # Exchange Transaction Charges (0.03503%)
      
    etc = exchange_txn_charges * (turnover)
    # SEBI Charges
    sebi_charges = sebi_charges_per * turnover

    # GST
    gst = 0.18 * (brokerage + etc)

    # Stamp Duty on buy side only
    stamp_duty = stamp_duty_per * buy_price * quantity

    # Total Charges
    total_charges = brokerage + stt + etc + gst + sebi_charges + stamp_duty

    # Gross Profit
    gross_profit = (sell_price - buy_price) * quantity

    # Net P&L
    net_pnl = gross_profit - total_charges

    # Points to breakeven
    points_to_breakeven = total_charges / quantity

    result = {
        'turnover': turnover,
        'brokerage': brokerage,
        'stt': stt,
        'exchange_txn_charges': etc,
        'sebi_charges': sebi_charges,
        'gst': gst,
        'stamp_duty': stamp_duty,
        'total_charges': total_charges,
        'gross_profit': gross_profit,
        'net_profit': net_pnl,
        'points_to_breakeven': points_to_breakeven,
    }

    return result


def convert_number_to_human_format(num: float, precision: int = 1) -> str:
    """
    Convert a number into human-readable format (e.g., 4.2k, 3.1m, 7.5b).
    
    Parameters:
    - num : float or int, the number to format
    - precision : int, number of decimal places
    
    Returns:
    - str : formatted number
    """
    if num is None:
        return "0"
    
    # Handle negatives
    sign = "-" if num < 0 else ""
    num = abs(num)
    
    # Define suffixes
    suffixes = ["", "k", "m", "b", "t"]
    idx = 0
    
    while num >= 1000 and idx < len(suffixes) - 1:
        num /= 1000.0
        idx += 1
    
    return f"{sign}{num:.{precision}f}{suffixes[idx]}"
