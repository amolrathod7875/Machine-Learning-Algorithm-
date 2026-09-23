import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

optimal_k = K[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters (by Silhouette Score): {optimal_k}")
print(f"Inertia at k={optimal_k}: {inertias[optimal_k - 2]:.2f}")
print(f"Silhouette Score at k={optimal_k}: {silhouette_scores[optimal_k - 2]:.4f}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

ari = adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index (vs true labels): {ari:.4f}")

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 3, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')

if optimal_k in [2, 3]:
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', edgecolor='k')
    plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
                s=300, c='red', marker='X', label='Centroids')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(f'K-Means Clusters (k={optimal_k})')
    plt.legend()

plt.tight_layout()
plt.savefig('elbow_and_clusters.png', dpi=150)
print("Plot saved as elbow_and_clusters.png")
