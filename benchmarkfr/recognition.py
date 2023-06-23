import logging
from typing import Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .logging import _

logger = logging.getLogger(__name__)


class KNN:

    def __init__(self, gallery_embeddings: np.ndarray, gallery_labels: np.ndarray, n_neighbors: int = 1):
        self.gallery_embeddings = gallery_embeddings
        self.gallery_labels = gallery_labels
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(self.n_neighbors, weights="uniform", algorithm="brute", n_jobs=None)
        self.model.fit(self.gallery_embeddings, self.gallery_labels)

        if self.gallery_labels.size < self.n_neighbors:
            raise ValueError("not enough minerals")

        logger.debug(_("Loaded KNN", n_neighbors=self.model.n_neighbors))

    def cluster_match(self, embeddings: np.ndarray) -> Tuple[int, float]:
        y_predict_np = self.model.predict(embeddings)
        # TO DO check KNN with a cluster of embeddings
        confidence = self.model.predict_proba(embeddings)
        confidence_max = np.max(confidence)

        # TODO : simplify the algorithm by adding a filter on the distance in the KNN

        # find the index of the max confidence
        index_max = np.argmax(confidence)
        # Vote inside the cluster
        # count the number of occurrence for each label
        idx_dict = dict(zip(set(y_predict_np), range(len(y_predict_np))))
        y_predict_np_int = [idx_dict[x] for x in y_predict_np]
        count_cluster = np.bincount(y_predict_np_int)
        idx_max = np.argmax(count_cluster)
        max_val = np.max(count_cluster)
        label_cluster = np.array([name for name, idx in idx_dict.items() if idx == idx_max])
        proba_label = max_val / sum(count_cluster)

        return label_cluster, proba_label
