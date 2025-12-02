import pandas as pd

def handle_outliers(df:pd.DataFrame,col,z_max = 3):
    """
    Removes outliers from a specified column in a DataFrame based on the Z-score method.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col : str
        The name of the column from which to remove outliers.
    z_max : int, optional
        The maximum absolute Z-score allowed. Values outside this range are considered outliers.
        Defaults to 3.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with outliers removed from the specified column.
    """
    
    df = df.copy()
    df['#z'] = ((df[col] - df[col].mean())/df[col].std())
    df = df[(df['#z'] < z_max) & (df['#z'] > -z_max)]
    df.drop(columns=['#z'],inplace=True)
    return df
