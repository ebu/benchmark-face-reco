import logging
import pathlib
from typing import Generator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(file_path: pathlib.Path, frame_rate: int = 1) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(str(file_path))

    if not cap.isOpened():
        raise FileNotFoundError

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Original frame size (width,height): ({frame_width},{frame_height})")
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Original frame rate: {fps:2f}")
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    logger.info(f"Original frame count: {frame_count}")

    frame_number = 0
    increment = round(fps / frame_rate)
    while frame_number < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if not success:
            break
        yield frame
        frame_number += increment

    cap.release()


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int, margin: int = 0) -> np.ndarray:
    x = max(x - margin // 2, 0)
    y = max(y - margin // 2, 0)
    width = width + margin
    height = height + margin
    return image[y:y + width, x:x + height, :]
