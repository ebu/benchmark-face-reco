import pathlib
import uuid
from typing import List, Tuple

import numpy as np
from numpy.testing import assert_array_equal

from benchmarkfr.face import BoundingBox, Face
from benchmarkfr.image import Image
from benchmarkfr.pipeline import run


def test_run_when_no_face_detected_then_no_face_group():
    # Arrange
    some_image = np.random.randint(0, 255, (480, 640, 3))

    def preprocess() -> List[pathlib.Path]:
        return [pathlib.Path("1.png")]

    def load_image(file_path: pathlib.Path) -> Image:
        return some_image

    def detect_faces(image: Image) -> List[Tuple[BoundingBox, float]]:
        assert_array_equal(image, some_image)
        return []

    def extract_embeddings(image: Image, bounding_box: BoundingBox) -> np.ndarray:
        raise Exception

    def extract_thumbnail(image: Image, bounding_box: BoundingBox) -> np.ndarray:
        return np.array([])

    def cluster_embeddings(embeddings: np.ndarray):
        assert_array_equal(embeddings, [])
        return []

    # Act
    face_groups = run(preprocess_fn=preprocess, load_image_fn=load_image, detect_faces_fn=detect_faces,
                      extract_embeddings_fn=extract_embeddings, extract_thumbnail_fn=extract_thumbnail,
                      cluster_embeddings_fn=cluster_embeddings)

    # Assert
    assert face_groups == []


def test_run_when_one_face_detected_then_one_face_group(mocker):
    # Arrange
    some_face_id = uuid.UUID(hex="43eaf4b780394c02b361370f83b0daad")
    some_bounding_box = BoundingBox(0, 0, 0, 0)
    some_confidence = 0.1
    some_embedding = np.random.random(128)
    some_thumbnail = np.array([])
    some_image_id = "1"
    some_image = np.random.randint(0, 255, (480, 640, 3))
    mocker.patch("uuid.uuid4").return_value = some_face_id
    some_face = Face(some_face_id.hex, some_bounding_box, some_confidence, some_embedding, some_thumbnail,
                     some_image_id)

    def preprocess() -> List[pathlib.Path]:
        return [pathlib.Path("1.png")]

    def load_image(file_path: pathlib.Path) -> Image:
        return some_image

    def detect_faces(image: Image) -> List[Tuple[BoundingBox, float]]:
        assert_array_equal(image, some_image)
        return [(some_bounding_box, some_confidence)]

    def extract_embeddings(image: Image, bounding_box: BoundingBox) -> np.ndarray:
        assert_array_equal(image, some_image)
        assert bounding_box == some_bounding_box
        return some_embedding

    def extract_thumbnail(image: Image, bounding_box: BoundingBox) -> np.ndarray:
        return some_thumbnail

    def cluster_embeddings(embeddings: np.ndarray):
        assert_array_equal(embeddings, [some_embedding])
        return [[0]]

    # Act
    face_groups = run(preprocess_fn=preprocess, load_image_fn=load_image, detect_faces_fn=detect_faces,
                      extract_embeddings_fn=extract_embeddings, extract_thumbnail_fn=extract_thumbnail,
                      cluster_embeddings_fn=cluster_embeddings)

    # Assert
    assert face_groups == [[some_face]]
