from pyannote.metrics.identification import IdentificationPrecision, IdentificationRecall, IdentificationErrorRate
from pyannote.core import Annotation, Segment
from typing import List
from benchmarkfr.face import FaceGroup
import pandas as pd

def gen_hyp_df(face_groups : List[FaceGroup]):
    # generate a dataframe with the detected celebrities from the pipeline output
    #predictions_celebs_format = [
    #     {'Timestamp': 0,  # in milliseconds
    #      'Celebrity': {
    #          'Urls': ['www.wikidata.org/wiki/Q19154', 'www.imdb.com/name/nm0636218'],
    #          # if you have it, can be useful to create celebrity ID
    #          'Name': 'Graham Norton'}  # similar to the video metadata (all_personalities field in the annotation file)
    #      },
    #     {'Timestamp': 1000,
    #      'Celebrity': {
    #          'Urls': ['www.wikidata.org/wiki/Q26876', 'www.imdb.com/name/nm2357847'],
    #          'Name': 'Taylor Swift'}
    #      },
    #     {'Timestamp': 1480,
    #      'Celebrity': {
    #          'Urls': ['www.wikidata.org/wiki/Q834621', 'www.imdb.com/name/nm0095104'],
    #          'Name': 'Bono'}},
    # ]
    #l.append(FaceGroup(uuid.uuid4().hex, person_id, confidence, faces)
    #Face(uuid.uuid4().hex, bounding_box, confidence, extract_embeddings_fn(image, bounding_box),
    #     extract_thumbnail_fn(image, bounding_box), file_path.stem)
    # we have a person ID with confidence for a face group
    # we need to create a dataframe with person_id, detected frame and name
    df = pd.DataFrame()
    # we define the columns
    df["person_id"] = None
    df["frame_idx"] = None
    # we fill the dataframe
    for face_group in face_groups:
        # we get the person ID
        person_id = face_group.person_id
        # we get the confidence
        confidence = face_group.confidence
        for face in face_group.faces:
            # we get the frame index
            frame_idx = face.file_path.stem
            # we get the name
            name = face.name
            # we add the row
            df.append({"person_id": person_id, "frame_idx": frame_idx, "name": name}, ignore_index=True)
    # then for each person ID we sort by frame index
    df.sort_values(by=["person_id", "frame_idx"], inplace=True, ascending=True)
    # we reset the index
    df.reset_index(inplace=True)
    return df

def get_intervals(df: pd.DataFrame):

    # get the name of the celebrities
    names = df["name"].unique()
    # for each name we iterate in frame index to build intervals
    # an interval is defined by a continuous sequence of frame index
    # where the name is detected
    intervals = {}
    for name in names:
        # we get the frame index where the name is detected
        frame_idxs = df[df["name"] == name]["frame_idx"].unique()
        # we sort the frame index
        frame_idxs.sort()
        # we iterate over the frame index
        for frame_idx in frame_idxs:
            # if the frame index is not in the intervals
            if frame_idx not in intervals:
                # we create a new interval
                intervals[frame_idx] = {"name": name, "start": frame_idx, "end": frame_idx}
            # if the frame index is in the intervals
            else:
                # we update the end of the interval
                intervals[frame_idx]["end"] = frame_idx
    # we create a dataframe with the intervals
    df_intervals = pd.DataFrame.from_dict(intervals, orient="index")
    # we sort the dataframe by start
    df_intervals.sort_values(by=["start"], inplace=True)
    # we reset the index
    df_intervals.reset_index(drop=True, inplace=True)
    return df_intervals










def eval_interval(reference: pd.DataFrame, hypothesis: pd.DataFrame):

    hypothesis_pya = Annotation()
    reference_pya = Annotation()

    for idx, row in hypothesis.iterrows():
        hypothesis_pya[Segment(row["start"], row["end"] + 1), idx] = row["name"]

    for idx, row in reference.iterrows():
        reference_pya[Segment(row["start"], row["end"] + 1), idx] = row["name"]

    return IdentificationPrecision()(reference_pya, hypothesis_pya), IdentificationRecall()(reference_pya,
                                                                                            hypothesis_pya)

    # TO DO :
    # metric per personality
    # metric per interval
