import pathlib

import numpy as np
import pytest

from benchmarkfr.image import extract_frames, resize_image


def test_extract_frames_when_file_does_not_exists():
    # Arrange

    # Act
    with pytest.raises(OSError):
        list(extract_frames(pathlib.Path("")))

    # Assert


def test_extract_frames():
    # Arrange

    # Act
    frames = list(extract_frames(pathlib.Path("data/videos/BA_GNS.mp4")))

    # Assert
    assert len(frames) == 59


@pytest.mark.skip
def test_extract_frames_():
    # Arrange

    # Act
    frames = list(extract_frames(pathlib.Path("data/videos/BA_GNS.mp4"), frame_rate=25))

    # Assert
    assert len(frames) == 1428


def test_resize_image():
    # Arrange
    frame = np.random.random((360, 640, 3))
    target_width = 320
    target_height = 180

    # Act
    resized_frame = resize_image(frame, (target_width, target_height))

    # Assert
    assert resized_frame.shape == (target_height, target_width, 3)
