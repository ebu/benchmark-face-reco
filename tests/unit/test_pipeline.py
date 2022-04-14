from typing import Any, Dict, List

import numpy as np

from benchmarkfr.pipeline import run


def test_run():
    # Arrange
    some_images = [np.random.randint(0, 255, (480, 640, 3))]

    def load_images() -> List[np.ndarray]:
        return some_images

    def detect_faces(image: np.ndarray) -> List[Dict[str, Any]]:
        return []

    def extract_embeddings(images: np.ndarray) -> np.ndarray:
        return np.random.random((len(images), 128))

    # Act
    images, image_numbers, bounding_boxes, confidences, embeddings = map(list, zip(*list(
        run(load_images=load_images, detect_faces=detect_faces, extract_embeddings=extract_embeddings))))

    # Assert
    assert images == some_images
    assert image_numbers == [[]]
    assert bounding_boxes == [[]]
    assert confidences == [[]]
    assert embeddings == [[]]
