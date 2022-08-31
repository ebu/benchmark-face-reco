import numpy as np

from benchmarkfr.recognition import KNN


def test_cluster_with_one_embedding():

    # Arrange
    embeddings = np.ones((1, 2))

    # Dictionary
    gallery_embeddings = np.ones((10, 2))
    gallery_labels = np.ones((10,))

    knn = KNN(gallery_embeddings, gallery_labels)

    # Act
    clusters = knn.cluster_match(embeddings)

    # Assert
    assert clusters == (1, 1.0)

def test_cluster_with_two_persons():
    # Arrange
    embeddings = np.ones((1, 3))

    # Dictionary
    gallery_embeddings = np.zeros((2,3))
    gallery_embeddings[0, :] = 1

    gallery_labels = np.array([0,1])

    n_neighbors = 1

    knn = KNN(gallery_embeddings, gallery_labels, n_neighbors)

    # Act
    clusters = knn.cluster_match(embeddings)

    # Assert
    assert clusters == (0, 1.0)


def test_cluster_with_two_similar_embeddings():
    # Arrange
    embeddings = np.ones((2, 2))

    # Act
    clusters = cluster(embeddings)

    # Assert
    assert len(clusters) == 1


def test_cluster_with_two_dissimilar_embeddings():
    # Arrange
    embeddings = np.array([np.ones(2), np.zeros(2)])

    # Act
    clusters = cluster(embeddings)

    # Assert
    assert len(clusters) == 2


def test_cluster_chinese_whispers_with_two_similar_embeddings():
    # Arrange
    embeddings = np.ones((2, 2))

    # Act
    clusters = cluster(embeddings, chinese_whispers)

    # Assert
    assert len(clusters) == 1


def test_cluster_chinese_whispers_with_two_dissimilar_embeddings():
    # Arrange
    embeddings = np.array([np.ones(2), np.zeros(2)])

    # Act
    clusters = cluster(embeddings, chinese_whispers)

    # Assert
    assert len(clusters) == 2


def test_cluster_hierarchical_with_two_similar_embeddings():
    # Arrange
    embeddings = np.ones((2, 2))

    # Act
    clusters = cluster(embeddings, hierarchical)

    # Assert
    assert len(clusters) == 1


def test_cluster_hierarchical_with_two_dissimilar_embeddings():
    # Arrange
    embeddings = np.array([np.ones(2), np.zeros(2)])

    # Act
    clusters = cluster(embeddings, hierarchical)

    # Assert
    assert len(clusters) == 2


def test_tsne():
    # Arrange
    embeddings = np.random.rand(37, 128)

    # Act
    points = tsne(embeddings)

    # Assert
    points.shape = (37, 2)
