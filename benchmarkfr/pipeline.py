import numpy as np
import pathlib
import uuid
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Callable, List, Tuple

import logging
from .face import BoundingBox, Face, FaceGroup
from .image import Image
from .log import _

logger = logging.getLogger(__name__)


def run(preprocess_fn: Callable[[], List[pathlib.Path]], load_image_fn: Callable[[pathlib.Path], Image],
        detect_faces_fn: Callable[[Image], List[Tuple[BoundingBox, float]]],
        extract_embeddings_fn: Callable[[Image, BoundingBox], np.ndarray],
        extract_thumbnail_fn: Callable[[Image, BoundingBox], np.ndarray],
        cluster_embeddings_fn: Callable[[np.ndarray], List[List[int]]],
        cluster_matching_fn: Callable[[np.ndarray], Tuple[int, float]]) -> List[List[Face]]:
    logger.info("Preprocessing")
    file_paths = preprocess_fn()

    faces = []
    #with logging_redirect_tqdm():
    for file_path in tqdm(file_paths):
        image = load_image_fn(file_path)
        logger.info(_("Loaded image", file_path=str(file_path)))

        logger.info(_("Detecting face(s)", file_path=str(file_path)))
        a = []
        for bounding_box, confidence in detect_faces_fn(image):
            logger.info("Extracting embedding")
            a.append(
                Face(uuid.uuid4().hex, bounding_box, confidence, extract_embeddings_fn(image, bounding_box),
                     extract_thumbnail_fn(image, bounding_box), file_path.stem))
        logger.info(_("Face(s) detected", file_path=str(file_path), face_ids=[face.id for face in a]))
        faces.extend(a)

    logger.info(_("Grouping face(s) embeddings", n_faces=len(faces)))
    face_groups = [[faces[index] for index in cluster] for cluster in
                   cluster_embeddings_fn(np.array([face.embedding for face in faces]))]

    logger.info(_("Face(s) embeddings grouped", n_face_groups=len(face_groups)))

    l = []
    for faces in face_groups:
        person_id, confidence = cluster_matching_fn(np.array([face.embedding for face in faces]))
        l.append(FaceGroup(uuid.uuid4().hex, person_id, confidence, faces))

    return l
