import logging
from typing import List, Tuple
from mtcnn import MTCNN as MTCNN_
from .face import BoundingBox
from .image import Image
from .log import _

logger = logging.getLogger(__name__)


class MTCNN:
    def __init__(self):
        self.model = MTCNN_()
        logger.debug(_("Loaded MTCNN", min_face_size=self.model.min_face_size))

    def detect_faces(self, image: Image) -> List[Tuple[BoundingBox, float]]:
        return [(BoundingBox(*face["box"]), face["confidence"]) for face in self.model.detect_faces(image)]
