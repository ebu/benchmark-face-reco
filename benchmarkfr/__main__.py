import argparse
import dataclasses
import json
import logging
import pathlib
from typing import List

import numpy as np
import shutil

from .clustering import cluster
from .facenet import Facenet
from .image import extract_frames, Image, load_image as load_image_, resize_image, save_image
from .logging import _, configure
from .mtcnn import MTCNN
from .pipeline import run

logger = logging.getLogger(__name__)


def load_image(file_path: pathlib.Path) -> Image:
    image = load_image_(file_path)
    target_size = (image.shape[1] // args.shrink_factor, image.shape[0] // args.shrink_factor)
    logger.debug(_("Resized image", file_path=str(file_path), image_size=(image.shape[1], image.shape[0]),
                   target_size=target_size))
    return resize_image(image, target_size)


def preprocess() -> List[pathlib.Path]:
    file_path = data_dir_path / "videos" / args.file_name
    logger.info(_("Extracting frame(s)", frame_rate=args.frame_rate, file_path=str(file_path)))
    dir_path = data_dir_path / "videos" / f"{pathlib.Path(args.file_name).stem}_{args.frame_rate}fps_RGB"
    shutil.rmtree(dir_path, ignore_errors=True)
    dir_path.mkdir()
    file_paths = []
    for i, frame in enumerate(extract_frames(file_path, args.frame_rate), start=1):
        file_paths.append(dir_path / f"{i}.png")
        save_image(file_paths[-1], frame)
        logger.debug(_("Frame saved", frame_num=i, file_path=str(file_paths[-1])))
    logger.info(_("Frame(s) extracted", n_frames=len(file_paths)))
    return file_paths


parser = argparse.ArgumentParser()
parser.add_argument("file_name")
parser.add_argument("--log-level", type=str, default="INFO")
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--frame-rate", type=int, default=1)
parser.add_argument("--shrink-factor", type=int, default=4)
args = parser.parse_args()

configure(args.log_level)

data_dir_path = pathlib.Path(args.data_dir)

result = run(preprocess_fn=preprocess, load_image_fn=load_image, detect_faces_fn=MTCNN().detect_faces,
             extract_embeddings_fn=Facenet(
                 model_path=data_dir_path / "models" / "facenet" / "keras" / "facenet_ds_keras_128.h5").extract_embeddings,
             cluster_embeddings_fn=cluster)


class CustomJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


print(json.dumps(result, cls=CustomJSONEncoder))
