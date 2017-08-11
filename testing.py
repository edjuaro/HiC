import pandas as pd
import sklearn.cluster as skl
# from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fclusterdata as cluster

df = pd.read_csv("test_dataset.gct", sep='\t', skiprows=2)
# print(df)
df.drop(['Name', 'Description'], axis=1, inplace=True)
df = df.T
print(df.shape)
# print(cluster(df, t=1.15))
# print(skl.AgglomerativeClustering(n_clusters=2, connectivity=df))
