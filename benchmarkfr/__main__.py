import argparse
import pathlib
from typing import List

import numpy as np
import shutil
from keras.models import load_model
from mtcnn import MTCNN

from .pipeline import run
from .preprocess import crop_image, extract_frames, resize_image


def save_results(dir_path: pathlib.Path, images: List[np.ndarray], image_numbers: List[List[int]],
                 bounding_boxes: List[List[List[int]]], confidences: List[List[float]],
                 embeddings: List[List[np.ndarray]]) -> None:
    shutil.rmtree(dir_path, ignore_errors=True)
    dir_path.mkdir()
    np.save(dir_path / "images", images)
    np.savez(dir_path / "faces", frame_numbers=image_numbers, bounding_boxes=bounding_boxes,
             confidences=confidences, embeddings=embeddings)


parser = argparse.ArgumentParser()
parser.add_argument("file_name")
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--frame-rate", type=int, default=1)
parser.add_argument("--shrink-factor", type=int, default=4)
args = parser.parse_args()

data_dir_path = pathlib.Path(args.data_dir)

save_results(data_dir_path / "arrays" / f"{pathlib.Path(args.file_name).stem}", *map(list, zip(*list(
    run(load_images=lambda: (
        resize_image(frame, (frame.shape[1] // args.shrink_factor, frame.shape[0] // args.shrink_factor)) for
        frame in extract_frames(data_dir_path / "videos" / args.file_name, args.frame_rate)),
        detect_faces=MTCNN().detect_faces,
        extract_embeddings=lambda image, bounding_box: load_model(
            data_dir_path / "models" / "facenet" / "keras" / "facenet_ds_keras_128.h5").predict_on_batch(
            resize_image(crop_image(image, *bounding_box), (160, 160))[None]))))))
