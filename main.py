import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# 1. Caricamento e Pre-processing
data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)

# 2. Riduzione della Dimensionalità (PCA)
# Riduciamo da 30 a 2 dimensioni per visualizzare i cluster
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Clustering con K-Means (Approccio Partizionale)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_km = kmeans.fit_predict(X_scaled)

# 4. Clustering con DBSCAN (Approccio basato sulla Densità)
dbscan = DBSCAN(eps=5, min_samples=5)
clusters_db = dbscan.fit_predict(X_scaled)

# 5. Valutazione Matematica
sil_km = silhouette_score(X_scaled, clusters_km)
# Nota: Calcoliamo la silhouette per DBSCAN solo se trova più di 1 cluster (escludendo il rumore)
mask = clusters_db != -1
sil_db = silhouette_score(X_scaled[mask], clusters_db[mask]) if len(np.unique(clusters_db[mask])) > 1 else 0

print(f"Silhouette Score K-Means: {sil_km:.4f}")
print(f"Silhouette Score DBSCAN: {sil_db:.4f}")

# 6. Visualizzazione Risultati
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot K-Means
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_km, cmap='viridis', alpha=0.6)
ax1.set_title(f'K-Means (Silhouette: {sil_km:.2f})')

# Plot DBSCAN
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_db, cmap='plasma', alpha=0.6)
ax2.set_title(f'DBSCAN (Silhouette: {sil_db:.2f})')

plt.savefig('images\clustering_comparison.png')
plt.show()