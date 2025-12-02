from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def kmeans_clustering(df, target, features, k, scaler=None, showGraph=True):
    """
    Perform K-Means clustering on selected features and analyze clusters based on a target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing both target and feature columns.
    target : str
        The target column used to interpret cluster meaning (not used in fitting).
    features : list of str
        The list of feature column names to use for clustering.
    k : int
        The number of clusters for K-Means.
    scaler : str or None, optional (default=None)
        Scaling method for features. Supported:
        - 'standard' : StandardScaler (zero mean, unit variance)
        - 'minmax'   : MinMaxScaler (range [0, 1])
        - 'robust'   : RobustScaler (robust to outliers)
        - None       : no scaling applied
    showGraph : bool, optional (default=True)
        If True, plot the clustering scatter plot.
    
    Returns
    -------
    df : pd.DataFrame
        Original DataFrame with an added column '{target}_cluster' for cluster labels.
    kmeans : sklearn.cluster.KMeans
        The fitted KMeans model.
    scaler : sklearn.preprocessing object or None
        The scaler used for transformation, or None if no scaling applied.
    
    Notes
    -----
    The `silhouette_score` measures how well-separated the clusters are.
    It ranges from -1 to 1:
        - +1 → clusters are well-separated
        - 0  → overlapping clusters
        - -1 → samples may be in wrong clusters
    """

    df = df.copy()
    df.dropna(subset=features, inplace=True)

    # --- Scaling ---
    scaler_obj = None
    X = df[features].values
    if scaler:
        if scaler == 'standard':
            scaler_obj = StandardScaler()
        elif scaler == 'minmax':
            scaler_obj = MinMaxScaler()
        elif scaler == 'robust':
            scaler_obj = RobustScaler()
        else:
            raise ValueError("Invalid scaler. Use 'standard', 'minmax', 'robust', or None.")
        X = scaler_obj.fit_transform(X)

    # --- KMeans ---
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df[f'{target}_cluster'] = kmeans.fit_predict(X)

    # --- Cluster sorting based on target mean ---
    cluster_means = df.groupby(f'{target}_cluster')[target].mean().sort_values()
    sorted_map = {old: new for new, old in enumerate(cluster_means.index)}
    df[f'{target}_cluster'] = df[f'{target}_cluster'].map(sorted_map).astype(int)

    # --- Cluster metrics ---
    score = silhouette_score(X, df[f'{target}_cluster'])
    cluster_stats = (
        df.groupby(f'{target}_cluster')[target]
          .agg(['min', 'max', 'mean'])
          .sort_values('mean')
    )

    print(f"\n📊 {target} Clustering Results:")
    print("Silhouette Score:", round(score, 4))
    print(cluster_stats)

    # --- Visualization ---
    if showGraph:
        plt.figure(figsize=(8, 4))
        plt.scatter(df.index, df[target], c=df[f'{target}_cluster'], cmap='tab10', s=25)
        plt.title(f"K-Means Clusters on {target}")
        plt.xlabel("Index")
        plt.ylabel(target)
        plt.show()

    return df, kmeans, scaler_obj




def evaluate_regression(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def train_models(df, target, features, scaler='robust', test_size=0.2, random_split=False):
    """
    If scaler == None  → create new StandardScaler()
    If scaler is an existing scaler → use it
    If scaler == False or 'none' → no scaling
    If random_split=True → use sklearn.train_test_split
    If random_split=False → use time-series split
    """

    df = df.copy().dropna(subset=features + [target])

    X = df[features].values
    y = df[target].values

    # ---------------- Scaling Logic ----------------
    scaler_obj = None
    if type(scaler) == str:
        if scaler == 'standard':
            scaler_obj = StandardScaler()
        elif scaler == 'minmax':
            scaler_obj = MinMaxScaler()
        elif scaler == 'robust':
            scaler_obj = RobustScaler()  
    elif isinstance(scaler, StandardScaler):
        scaler_obj = scaler
        X = scaler_obj.transform(X)
    X = scaler_obj.fit_transform(X)
    # ---------------- Splitting Logic ----------------
    if random_split:
        # random split with shuffling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=42
        )
    else:
        # time-ordered split (no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

    results = {}

    # ---------------- Linear Regression ----------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['LinearRegression'] = {
        'model': lr, 'pred': y_pred_lr, 'eval': evaluate_regression(y_test, y_pred_lr),'actual': y_test
    }

    # ---------------- Ridge Regression ----------------
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results['Ridge'] = {
        'model': ridge, 'pred': y_pred_ridge, 'eval': evaluate_regression(y_test, y_pred_ridge),'actual': y_test
    }

    # ---------------- Random Forest ----------------
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['RandomForest'] = {
        'model': rf, 'pred': y_pred_rf, 'eval': evaluate_regression(y_test, y_pred_rf),'actual': y_test
    }

    # ---------------- XGBoost ----------------
    try:
        xgb = XGBRegressor(
            n_estimators=200, max_depth=6, random_state=42, verbosity=0
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        results['XGBoost'] = {
            'model': xgb, 'pred': y_pred_xgb, 'eval': evaluate_regression(y_test, y_pred_xgb),'actual': y_test
        }
    except Exception as e:
        print("⚠️ XGBoost not available:", e)

    return results, scaler_obj


### Neural ODE Model ###
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, x):
        return self.net(x)

def train_neural_ode(df, target, features, scaler='robust', test_size=0.2, random_split=False,batch_size=32, epochs=40, lr=0.001, hidden_dim=64):

    df = df.copy().dropna(subset=features + [target])

    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)

    scaler_obj = None
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    elif scaler == "robust":
        scaler_obj = RobustScaler()
    elif scaler is None or scaler is False:
        scaler_obj = None
    else:
        raise ValueError("Scaler must be 'standard', 'minmax', 'robust', None, or False")

    if scaler_obj is not None:
        X = scaler_obj.fit_transform(X)

    # ----------- Splitting Logic -----------
    if random_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=42
        )
    else:
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

    # Convert to torch
    X_train_torch = torch.tensor(X_train)
    y_train_torch = torch.tensor(y_train)

    dataset = TensorDataset(X_train_torch, y_train_torch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ----------- Neural ODE Model -----------
    input_dim = X_train.shape[1]

    func = ODEFunc(input_dim=input_dim, hidden_dim=hidden_dim)
    ode_model = func

    optimizer = torch.optim.Adam(ode_model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Time horizon: short (ODE behaves like deep residual)
    t = torch.tensor([0.0, 1.0])

    # ----------- Training Loop -----------
    
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            # Integrate ODE
            pred = odeint(ode_model, batch_x, t)[1]

            # Linear map to target
            pred_target = nn.Linear(input_dim, 1)(pred)

            loss = mse(pred_target, batch_y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.5f}")

    # ----------- Prediction -----------
    X_test_torch = torch.tensor(X_test)
    y_pred = odeint(ode_model, X_test_torch, t)[1]
    y_pred = nn.Linear(input_dim, 1)(y_pred).detach().cpu().numpy()

    # Flatten
    y_pred_flat = y_pred.reshape(-1)
    y_test_flat = y_test.reshape(-1)

    # ----------- Evaluation + Return -----------

    results = {}
    results["ode"] = {
        'model': ode_model,
        'pred': y_pred_flat,
        'actual': y_test_flat,
        'eval': evaluate_regression(y_test_flat, y_pred_flat)
    }

    return results, scaler_obj