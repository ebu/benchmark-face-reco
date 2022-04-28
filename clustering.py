from typing import List

import dlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.cm import get_cmap
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN as DBSCAN_
from sklearn.manifold import TSNE


def tsne(embeddings: np.ndarray) -> np.ndarray:
    return TSNE(init="pca", random_state=0).fit_transform(embeddings)


def plot(points: np.ndarray, clusters: np.ndarray) -> None:
    plt.figure(figsize=(10, 10))
    cmap = get_cmap("tab20c")
    colors = cmap.colors
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    for i, cluster in enumerate(clusters, start=1):
        plt.scatter(x=points[np.array(cluster), 0], s=16, y=points[np.array(cluster), 1], label=i)
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.legend(bbox_to_anchor=(1.02, 1.01))
    plt.show()


def DBSCAN(embeddings: np.ndarray, min_samples=1, eps=0.5) -> List[int]:
    return DBSCAN_(min_samples=min_samples, eps=eps).fit(embeddings).labels_


def chinese_whispers(embeddings: np.ndarray, threshold=1) -> List[int]:
    embeddings = [dlib.vector(embedding) for embedding in embeddings]
    return dlib.chinese_whispers_clustering(embeddings, threshold)


def hierarchical(embeddings: np.ndarray, distance_threshold=1) -> List[int]:
    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity="euclidean",
                                      linkage="ward")
    return cluster.fit_predict(embeddings)


def cluster(embeddings: np.ndarray, method=DBSCAN, **kwargs) -> List[List[int]]:
    if not embeddings.size:
        return []
    labels = method(embeddings, **kwargs)
    return [np.asarray(labels == label).nonzero()[0].tolist() for label in np.unique(labels)]
