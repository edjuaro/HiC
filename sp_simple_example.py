import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import linkage
from time import time
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
import pandas as pd
from src.ccalnoir.ccalnoir.mathematics.information import information_coefficient_nonneg

# a custom function that just computes Euclidean distance
def mydist(p1, p2):
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

print(X)
print(X.shape)
print(y)
print(y.shape)
print('-----')
print("Loading GP's simple dataset.")
df = pd.read_csv("test_dataset.gct", sep='\t', skiprows=2)
df.drop(['Name', 'Description'], axis=1, inplace=True)
data = df.as_matrix().T
print(data, data.shape)
df = pd.read_csv("test_dataset.cls", sep=' ', skiprows=2, header=None)
labels = np.asarray(df.as_matrix().T)
labels = labels.reshape(labels.shape[0],)
print(labels, labels.shape)

np.random.seed(0)

# ----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    # for i in range(X_red.shape[0]):
    for i in range(X_red.shape[0]-1):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

# ----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
# X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
data_red = manifold.SpectralEmbedding(n_components=2).fit_transform(data)
print("Done.")

t0 = time()
#fclust = fclusterdata(data, t=3, criterion="inconsistent", metric=mydist)
#fclust2 = fclusterdata(data, t=3, criterion="inconsistent", metric=information_coefficient_nonneg)

fclust = fclusterdata(X, t=2, criterion="inconsistent", metric=mydist)
fclust2 = fclusterdata(X, t=2, criterion="inconsistent", metric=information_coefficient_nonneg)

#link = linkage(X_red, metric=mydist)
#link2 = linkage(data, metric=mydist)
print("took {} seconds".format(time() - t0))

print(fclust)
print(fclust2)
print(labels)
#print(link2)

#plot_clustering(X_red, X, link, "linkage")
#plot_clustering(X_red, X, link2, "linkage")
#plt.show()

