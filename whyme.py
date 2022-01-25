# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:04:04 2022

@author: User
"""

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
wine_original=pd.read_csv('./data/wines.csv')
# Examine data
wine_original.head()

wine_original
wine_df = wine_original.drop(columns="Region")
wine_df
scaler = preprocessing.StandardScaler().fit(wine_df)
scaler
#Z-transform


wine_df_scaled = pd.DataFrame(scaler.fit_transform(wine_df), columns=wine_df.columns, index=wine_df.index)
wine_df_scaled
cluster_range = range(1, 10)
cluster_errors = []
for num_clusters in cluster_range:
    clusters=KMeans(num_clusters)
    clusters.fit(wine_df_scaled)
    cluster_errors.append(clusters.inertia_)
    
plt.figure(figsize=(6,4))
plt.plot(cluster_range,cluster_errors,marker="o")
plt.show()

k=3
model=KMeans(n_clusters=k,random_state=10).fit(wine_df_scaled)
print(model)
print(model.labels_)
centroids=model.cluster_centers_