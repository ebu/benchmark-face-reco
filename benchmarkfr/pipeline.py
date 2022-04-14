from typing import Any, Callable, Dict, Generator, List, Tuple

import numpy as np


def run(load_images, detect_faces: Callable[[np.ndarray], List[Dict[str, Any]]],
        extract_embeddings: Callable[[np.ndarray, List[List[int]]], np.ndarray]) -> Generator[
    Tuple[np.ndarray, List[int], List[List[int]], List[float], List[np.ndarray]], None, None]:
    for image_number, image in enumerate(load_images()):
        image_numbers, bounding_boxes, confidences, embeddings = ([],) * 4
        for face in detect_faces(image):
            image_numbers.append(image_number)
            bounding_boxes.append(face["box"])
            confidences.append(face["confidence"])
            embeddings.append(extract_embeddings(image, face["box"]))
        yield image, image_numbers, bounding_boxes, confidences, embeddings
