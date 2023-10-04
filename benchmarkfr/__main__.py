import argparse
import dataclasses
import json
import pathlib
import shutil
import logging
import numpy as np

from typing import List

from .clustering import cluster, hierarchical
from .face import BoundingBox
from .embeddings import Facenet
from .image import crop_image, extract_frames, Image, load_image as load_image_, resize_image, save_image
from .detection import MTCNN
from .pipeline import run
from .recognition import KNN
from .log import _, configure
from .metrics import video_eval

logger = logging.getLogger(__name__)

def load_image(file_path: pathlib.Path) -> Image:
    image = load_image_(file_path)
    if args.shrink_factor == 1:
        return image
    target_size = (image.shape[1] // args.shrink_factor, image.shape[0] // args.shrink_factor)
    return resize_image(image, target_size)


def preprocess() -> List[pathlib.Path]:
    file_path = data_dir_path / "videos" / args.file_name
    logger.info(_("Extracting frame(s)", frame_rate=args.frame_rate, file_path=str(file_path)))
    dir_path = data_dir_path / "videos" / f"{pathlib.Path(args.file_name).stem}_{args.frame_rate}fps_RGB"
    shutil.rmtree(dir_path, ignore_errors=True)
    dir_path.mkdir()
    file_paths = []
    for i, frame in enumerate(extract_frames(file_path, args.frame_rate)):
        file_paths.append(dir_path / f"{i}.png")
        save_image(file_paths[-1], frame)
        logger.debug(_("Frame saved", frame_num=i, file_path=str(file_paths[-1])))
    logger.info(_("Frame(s) extracted", n_frames=len(file_paths)))

    return file_paths


def zero_shot_classifier():
    embeddings_file_path = data_dir_path / "embeddings" / "gallery_embeddings.npz"
    a = np.load(embeddings_file_path)  # gallery embeddings
    return KNN(a["embeddings"], a["person_id"]).cluster_match


def extract_thumbnail(image: Image, bounding_box: BoundingBox) -> np.ndarray:
    return resize_image(crop_image(image, bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height, 10),
                        (224, 224))


parser = argparse.ArgumentParser()
parser.add_argument("file_name")
parser.add_argument("--log-level", type=str, default="INFO")
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--frame-rate", type=int, default=1)
parser.add_argument("--shrink-factor", type=int, default=4)
parser.add_argument("--eval", action='store_true')
args = parser.parse_args()

configure(args.log_level)

data_dir_path = pathlib.Path(args.data_dir)

result = run(preprocess_fn=preprocess, load_image_fn=load_image, detect_faces_fn=MTCNN().detect_faces,
             extract_embeddings_fn=Facenet(
                 model_path=data_dir_path / "models" / "facenet" / "keras" / "facenet_ds_keras_512.h5").extract_embeddings,
             extract_thumbnail_fn=extract_thumbnail,
             cluster_embeddings_fn=lambda embeddings: cluster(embeddings, hierarchical),
             cluster_matching_fn=zero_shot_classifier())

class CustomJSONEncoder(json.JSONEncoder):

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

if args.eval:
    logger.info("Evaluation")
    eval_report = video_eval(data_dir_path, pathlib.Path(args.file_name).stem, result)
    
    with open(data_dir_path / "results" / f"report_{pathlib.Path(args.file_name).stem}.json", 'w') as f:
        f.write(json.dumps(eval_report, cls=CustomJSONEncoder, indent=4))
    
print(json.dumps(result, cls=CustomJSONEncoder))

#with open(data_dir_path / "results" / f"{pathlib.Path(args.file_name).stem}.json", 'w') as f:
#    f.write(json.dumps(result, cls=CustomJSONEncoder, indent=4))

