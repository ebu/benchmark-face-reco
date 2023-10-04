import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Face:
    id: str
    bounding_box: BoundingBox
    confidence: float
    embedding: np.ndarray
    thumbnail: np.ndarray
    image_id: str


@dataclass
class FaceGroup:
    id: str
    person_id: List[str]
    confidence: float
    faces: List[Face]