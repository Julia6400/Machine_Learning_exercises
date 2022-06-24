import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
seed = np.random.seed(204)
from sklearn.datasets import load_wine
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

wine = load_wine()
dir(wine)

X, y = wine.data, wine.target
scaler = StandardScaler().fit(X)
X_std = scaler.transform(X)

elements = [2, 4, 6]
ele_sum = []

for ele in elements:
    pca = PCA(n_components=ele, random_state=seed)
    pca.fit(X_std)

    expl_ratio = pca.explained_variance_ratio_
    print(expl_ratio)

    expl_ratio_sum = sum(expl_ratio)
    ele_sum.append(expl_ratio_sum)

df_to_plot = pd.DataFrame(ele_sum, index=elements, columns=["expl_ratio"])
df_to_plot.index.name = "n. of components"
df_to_plot.plot(kind="bar", grid=True)

pca = PCA(n_components=0.95, random_state=seed)
x_reduced = pca.fit_transform(X)

pca.n_components_

train_data = pd.read_csv("data/train_data.csv", header=None)
test_data = pd.read_csv("data/test_data.csv", header=None)

train_data.info()
test_data.info()

train_data_02, train_data_08 = train_test_split(train_data, test_size=0.8, shuffle=True, random_state=seed)
test_data_02, test_data_08 = train_test_split(test_data, test_size=0.8, shuffle=True, random_state=seed)

scaler = StandardScaler().fit(test_data_02)
data_std = scaler.transform(test_data_02)

pca = PCA(n_components=2, whiten=True, random_state=seed)
pca_data = pca.fit_transform(data_std)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

tsne = TSNE(n_components=2, random_state=seed)
tsne_data = tsne.fit_transform(data_std)

print(tsne.kl_divergence_)

data_sets = [pca_data, tsne_data]
data_names = ["pca_data", "tsne_data"]
colors = ["r", "g", "lb"]

for i, n, c in zip(data_sets, data_names, colors):
    plt.figure(figsize=(8, 6))
    start_time = time.time()
    plt.scatter(
        i[:, 0], i[:, 1],
        s=75, c=c,
        marker="o",
        alpha=0.5,
        edgecolor="black"
    )
    end_time = time.time()
    plt.title(n)
    print(f"time of 1 plot: {end_time - start_time} seconds")

pipe = Pipeline([
    ("std", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=seed)),
    ("tsne", TSNE(n_components=2, random_state=seed)),
])

piped = pipe.fit_transform(test_data_02)

plt.figure(figsize=(8, 6))
plt.scatter(
    piped[:, 0],
    piped[:, 1],
    s=75, c="b",
    marker="o",
    alpha=0.6,
    edgecolor="black"
    )

plt.title(data_names)
plt.grid()
plt.show()

train_labels = pd.read_csv("data/train_labels.csv", header=None)
y2 = train_labels[0].values
X_train, X_test, y_train, y_test = train_test_split(train_data, y2, test_size=0.95, shuffle=True, random_state=seed)
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)

kpca = KernelPCA()
svc_model = SVC()

pipe = Pipeline(steps=[('kpca', kpca), ('svc_model', svc_model)])
pipe.fit(X_train_std, y_train)

param_dict = {"kpca__gamma": np.linspace(0.03, 0.05, 5),
              "kpca__kernel": ['linear', 'poly', 'rbf'],
              "svc_model__C": [0.1, 1, 10, 100, 1000],
              "svc_model__gamma": [1, 0.1, 0.01, 0.001, 0.0001]}


grid = GridSearchCV(pipe, param_dict, verbose=0)
grid.fit(X_train_std, y_train)
grid.best_params_
grid.best_score_