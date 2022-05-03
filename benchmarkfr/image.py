import pathlib
from typing import Iterator, Tuple

import cv2
import imageio.v3 as iio
import numpy as np

Image = np.ndarray


def extract_frames(file_path: pathlib.Path, frame_rate: int = 1) -> Iterator[Image]:
    return iio.imiter(file_path, fps=frame_rate)


def load_image(file_path: pathlib.Path) -> Image:
    return iio.imread(file_path)


def save_image(file_path: pathlib.Path, image: Image):
    iio.imwrite(file_path, image)


def resize_image(image: Image, target_size: Tuple[int, int]) -> Image:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def crop_image(image: Image, x: int, y: int, width: int, height: int, margin: int = 0) -> Image:
    x = max(x - margin // 2, 0)
    y = max(y - margin // 2, 0)
    width = width + margin
    height = height + margin
    return image[y:y + width, x:x + height, :]
