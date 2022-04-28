import numpy as np

from clustering import cluster, chinese_whispers, hierarchical, tsne


def test_cluster_without_embeddings():
    # Arrange
    embeddings = np.array([])

    # Act
    clusters = cluster(embeddings)

    # Assert
    assert clusters == []


def test_cluster_with_one_embedding():
    # Arrange
    embeddings = np.ones((1, 2))

    # Act
    clusters = cluster(embeddings)

    # Assert
    assert clusters == [[0]]


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
