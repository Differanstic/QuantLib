from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from . import CatDB
from . import data_loader


def pipelineRegimeDetection(df,date,kMeansPath,scalerPath,xCols,labels,tf='15min'):
    testDB = data_loader.loadNifty(date)
    testCatDF = CatDB.indexCatDB(testDB,tf)
    
    kmeansModel = load(kMeansPath)
    scaler = load(scalerPath)
    
    X = testCatDF[xCols].values
    X_scaled = scaler.fit_transform(X)
    testCatDF['Regime'] = kmeansModel.predict(X_scaled)
    
    
    #centroids = pd.DataFrame(scaler.inverse_transform(kmeansModel.cluster_centers_),columns=xCols)
    #centroids['cluster'] = range(len(labels))
    #mapping = dict(zip(centroids['cluster'], centroids['regime_label']))
    #testCatDF['regime_label'] = testCatDF['Regime'].map(mapping)
    testCatDF_reset = testCatDF.reset_index().rename(columns={'index':'timestamp'})
    testCatDF_reset['timestamp'] = pd.to_datetime(testCatDF_reset['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = pd.merge_asof(
        df,
        testCatDF_reset[['timestamp', 'Regime']].sort_values('timestamp'),
        on='timestamp'
    )
    return df
