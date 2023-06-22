import numpy as np
import pathlib
from keras.models import load_model
from sklearn.preprocessing import normalize
from typing import Tuple

import logging
from .face import BoundingBox
from .image import crop_image, Image, resize_image
from .logging import _

logger = logging.getLogger(__name__)

def l2_normalize_512(img):
    return  (np.float32(img) - 127.5) / 128.0

class Facenet:
    def __init__(self, model_path: pathlib.Path, target_size: Tuple[int, int] = (160, 160)):
        self.model = load_model(model_path)
        logger.debug(_("Loaded Facenet"))
        self.target_size = target_size

    def extract_embeddings(self, image: Image, bounding_box: BoundingBox) -> np.ndarray:
        return normalize(self.model.predict_on_batch(
            l2_normalize_512(resize_image(crop_image(image, bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height),
                         self.target_size)[None]))).squeeze()
