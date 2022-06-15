import numpy as np
from dataclasses import dataclass


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
