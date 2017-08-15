# Authors: Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2014

# print(__doc__)
from time import time
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import scipy

from sklearn import manifold, datasets
from src.ccalnoir.ccalnoir.mathematics.information import information_coefficient
import pandas as pd


df = pd.read_csv("test_dataset.gct", sep='\t', skiprows=2)
dat = pd.read_csv("test_dataset.gct", sep='\t', skiprows=2)
# df = pd.read_csv("all_aml_test.gct", sep='\t', skiprows=2)
df.drop(['Name', 'Description'], axis=1, inplace=True)
data = df.as_matrix().T
# print(data, data.shape)
df = pd.read_csv("test_dataset.cls", sep=' ', skiprows=2, header=None)
# df = pd.read_csv("all_aml_test.cls", sep=' ', skiprows=2, header=None)
new_labels = np.asarray(df.as_matrix().T)
new_labels = new_labels.reshape(new_labels.shape[0],)

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

np.random.seed(0)


# a custom function that just computes Euclidean distance
def mydist(p1, p2):
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5


def custom_pearson(x, y):
    return scipy.stats.pearsonr(x, y)[0]


def myaffintyi(M):
    return np.array([[information_coefficient(a, b) for a in M]for b in M])


def myaffintye(M):
    return np.array([[mydist(a, b) for a in M]for b in M])


def myaffintyp(M):
    return np.array([[custom_pearson(a, b) for a in M]for b in M])


def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y

# X, y = nudge_images(X, y)
# data, new_labels = nudge_images(data, new_labels)

# ----------------------------------------------------------------------


# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(new_labels[i]),
                 color=plt.cm.spectral(labels[i] / 2.),
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


def count_diff(x):
    count = 0
    compare = x[0]
    for i in x:
        if i != compare:
            count += 1
    return count


func_dic = {
    myaffintye: "custom_eucledian",
    myaffintyi: "custom_ic",
    myaffintyp: "custom_pearson",
    'l1': 'l1',
    'l2': 'l2',
    'manhattan': 'manhattan',
    'cosine': 'cosine',
}


def count_mislabels(labels, true):
    return count_diff(labels[:21]) + count_diff(labels[21:])

from sklearn.cluster import AgglomerativeClustering

for affinity in [myaffintye, myaffintyp, myaffintyi, 'l1', 'l2', 'manhattan', 'cosine']:
    clustering = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=affinity)
    t0 = time()
    clustering.fit(data)
    # clustering.fit(X)
    # print(new_labels)
    # print(clustering.labels_)
    print("%s : %.2fs : %i errors" % (func_dic[affinity], time() - t0, count_mislabels(clustering.labels_, new_labels)))

    plot_clustering(data_red, data, clustering.labels_, "Affinity/metric = {}".format(func_dic[affinity]))

    plt.savefig('af-'+func_dic[affinity]+'.png', dpi=300)



# import seaborn as sn
# dat.set_index(dat['Name'], inplace=True)
# dat.drop(['Name', "Description"], axis=1, inplace=True)
# print(dat)
# g = sn.clustermap(dat.T)
# plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
# plt.show()
