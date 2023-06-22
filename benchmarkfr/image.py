import cv2
import imageio.v3 as iio
import logging
import numpy as np
import pathlib
from typing import Iterator, Tuple

from .logging import _

Image = np.ndarray

logger = logging.getLogger(__name__)


def extract_frames(file_path: pathlib.Path, frame_rate: int = 1) -> Iterator[Image]:
    return iio.imiter(file_path, fps=frame_rate)


def load_image(file_path: pathlib.Path) -> Image:
    return iio.imread(file_path, mode="RGB")


def save_image(file_path: pathlib.Path, image: Image):
    iio.imwrite(file_path, image)


def resize_image(image: Image, target_size: Tuple[int, int]) -> Image:
    logger.debug(_("Resized image", image_size=(image.shape[1], image.shape[0]),
                   target_size=target_size))
    if image.shape[0] < target_size[1] or image.shape[1] < target_size[0]:
        logger.warning("Upscaling image")
    return cv2.resize(image, target_size)


def crop_image(image: Image, x: int, y: int, width: int, height: int, margin: int = 35) -> Image:
    x = max(x - margin // 2, 0)
    y = max(y - margin // 2, 0)
    width = width + margin
    height = height + margin
    return image[y:y + height, x:x + width, :]
