from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __str__(self):
        return f"[x={self.x}, y={self.y}, width={self.width}, height={self.height}]"


@dataclass
class Face:
    id: str
    bounding_box: BoundingBox
    confidence: float
    embedding: np.ndarray
    image_id: str

    def __str__(self):
        return f"[id={self.id}, bounding_box={str(self.bounding_box)}, confidence={self.confidence}]"
