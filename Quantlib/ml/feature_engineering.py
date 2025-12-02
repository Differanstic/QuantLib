import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def time_sin_cos(df,time_col = None):
    """
    Adds sine and cosine transformations of time to capture cyclical patterns.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing a time column.
    time_col (str): Name of the time column in df. If None, uses the index.

    Returns:
    pd.Series: Sin and Cos transformations of time.
    """
    if time_col:
        time_data = pd.to_datetime(df[time_col])
        time_in_seconds = (time_data.dt.hour * 3600 + time_data.dt.minute * 60 + time_data.dt.second)
    else:
        time_data = pd.to_datetime(df.index)
        time_in_seconds = (time_data.hour * 3600 + time_data.minute * 60 + time_data.second)

    seconds_in_day = 24 * 60 * 60
    
    
    sin = np.sin(2 * np.pi * time_in_seconds / seconds_in_day)
    cos = np.cos(2 * np.pi * time_in_seconds / seconds_in_day)
    
    return sin, cos    




def feature_importance(results, features):
    """
    Extracts feature importance from trained models in `results`.

    Parameters
    ----------
    results : dict
        The output of train_models(): results['model'] must exist.
    features : list
        List of feature names used in training.

    Returns
    -------
    importance_dict : dict
        { model_name : pd.DataFrame(Feature, Importance) }
    """

    importance_dict = {}

    for model_name, data in results.items():
        model = data["model"]

        fi = None

        # ------------- Linear & Ridge (coefficients) -------------
        if isinstance(model, (LinearRegression, Ridge)):
            if hasattr(model, "coef_"):
                coefs = np.abs(model.coef_).flatten()
                fi = pd.DataFrame({
                    "Feature": features,
                    "Importance": coefs / coefs.sum()  # normalized
                }).sort_values("Importance", ascending=False)

        # ------------- Random Forest -------------
        elif isinstance(model, RandomForestRegressor):
            fi_raw = model.feature_importances_
            fi = pd.DataFrame({
                "Feature": features,
                "Importance": fi_raw / fi_raw.sum()
            }).sort_values("Importance", ascending=False)

        # ------------- XGBoost -------------
        elif isinstance(model, XGBRegressor):
            fi_raw = model.feature_importances_
            fi = pd.DataFrame({
                "Feature": features,
                "Importance": fi_raw / fi_raw.sum()
            }).sort_values("Importance", ascending=False)

        # ------------- Unsupported models -------------
        else:
            print(f"⚠️ Model '{model_name}' does not support feature importance.")
            continue

        importance_dict[model_name] = fi

    return importance_dict