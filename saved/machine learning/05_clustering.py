from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
seed = np.random.seed(61)

wine = load_wine()
dir(wine)
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# standar scaling
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# dimensionality reduction
pca = PCA(n_components=2, whiten=True)
X_reducted = pca.fit_transform(X_train_std)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

# scatter for claster visibility
plt.figure(figsize=(8, 6))
plt.scatter(
    X_reducted[:, 0],
    X_reducted[:, 1],
    s=100
)

plt.show()



# k means determine k
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

range_n_clusters = list(range(2, 10))
silhouette_score = []

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = metrics.silhouette_score(X, cluster_labels)
    silhouette_score.append(silhouette_avg)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

plt.plot(range_n_clusters, silhouette_score, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette_score')
plt.title('The Silhouette Coefficient method showing the optimal k')
plt.show()


km_cls = KMeans(
    n_clusters=2,
    init='random',
    n_init=10,
    max_iter=300,
    random_state=seed
)

y_km = km_cls.fit_predict(X_reducted)
km_cls.cluster_centers_

plt.figure(figsize=(10, 8))

plt.scatter(
    X_reducted[y_km == 0, 0],
    X_reducted[y_km == 0, 1],
    c='green', edgecolor='black',
    label='1', s=80
)

plt.scatter(
    X_reducted[y_km == 1, 0],
    X_reducted[y_km == 1, 1],
    c='orange', edgecolor='black',
    label='1', s=80
)

plt.scatter(
    km_cls.cluster_centers_[:, 0],
    km_cls.cluster_centers_[:, 1],
    c='r', marker="o", s=200, alpha=0.6,
    edgecolor="black", label='Cent'
)

plt.legend()
plt.grid()
plt.show()


init_list = ["random", "k-means++"]

for i in init_list:
    model = KMeans(
        n_clusters=2,
        init=i,
        n_init=10,
        max_iter=300,
        random_state=seed
    )

    model.fit(X, y)
    pred = model.predict(X)
    print(f"Kmeans with init= {i}. ACC: {accuracy_score(y, pred)}")


# Compute clustering with MeanShift

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# Plot result
plt.figure(figsize=(10, 8))
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()