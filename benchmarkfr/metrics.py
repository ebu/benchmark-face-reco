import json
import pathlib
import cv2
import pandas as pd

from pyannote.metrics.identification import IdentificationPrecision, IdentificationRecall, IdentificationErrorRate
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.utils import UEMSupportMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from datetime import timedelta, datetime
from ast import literal_eval
from typing import List

from .face import FaceGroup



def generate_all_intervals(annotation_json: dict, nb_frames: int, fps: int) -> pd.DataFrame:
    """
    From reference json file, generate missing intervals without personality
    """
    
    annotation_df = pd.DataFrame.from_dict(annotation_json, orient='index')

    # add a first empty real interval if the first annotation interval does not start at 0
    result_dict = {}
    first_frames = literal_eval(annotation_df.iloc[0].frame_interval)
    if first_frames[0] > 0:
        first_times = str((datetime.strptime(annotation_df.iloc[0].time_interval.split(',')[0][1:],"%H:%M:%S.%f") - timedelta(seconds=float(annotation_df.iloc[0].time_interval.split(',')[2][:-1]))).time())
        result_dict['new_first'] = {
            'time_interval': ['00:00:00.000', first_times+'.000',annotation_df.iloc[0].time_interval.split(',')[2][:-1]],
            'frame_interval':  [0, first_frames[0]-fps, fps],
            'personalities': [],
        }

    # add empty intermediate intervals if absent between two with personalities 
    for i in range(annotation_df.shape[0]-1):
        time_interval_list = annotation_df.iloc[i]['time_interval'].split(',')
        result_dict[str(i)] = {
            'time_interval': [time_interval_list[0][1:],time_interval_list[1],time_interval_list[2][:-1]],
            'frame_interval':  literal_eval(annotation_df.iloc[i]['frame_interval']),
            'personalities': annotation_df.iloc[i]['personalities'],
        }
        first_frame = literal_eval(annotation_df.iloc[i].frame_interval)
        second_frame = literal_eval(annotation_df.iloc[i+1].frame_interval)

        if second_frame[0]-fps > first_frame[1]:
            start_times = str((datetime.strptime(annotation_df.iloc[i].time_interval.split(',')[1],"%H:%M:%S.%f") + timedelta(seconds=float(annotation_df.iloc[i].time_interval.split(',')[2][:-1]))).time())
            end_times = str((datetime.strptime(annotation_df.iloc[i+1].time_interval.split(',')[0][1:],"%H:%M:%S.%f") - timedelta(seconds=float(annotation_df.iloc[i+1].time_interval.split(',')[2][:-1]))).time())
            result_dict[f'new_{i}_{i+1}'] = {
                'time_interval': [start_times+'.000', end_times+'.000',time_interval_list[2][:-1]],
                'frame_interval':  [first_frame[1]+fps, second_frame[0]-fps, fps],
                'personalities': [],
            }
        time_interval_list = annotation_df.iloc[i+1]['time_interval'].split(',')
        result_dict[str(i+1)] = {
            'time_interval': [time_interval_list[0][1:],time_interval_list[1],time_interval_list[2][:-1]],
            'frame_interval':  literal_eval(annotation_df.iloc[i+1]['frame_interval']),
            'personalities': annotation_df.iloc[i+1]['personalities'],
        }

    # add an empty last real interval if the last annotation interval does not end at the last frame
    last_frames = literal_eval(annotation_df.iloc[-1].frame_interval)
    real_last_frame = list(range(0,nb_frames+1,fps))[-1]

    if last_frames[1] < real_last_frame:
        last_times = str((datetime.strptime(annotation_df.iloc[-1].time_interval.split(',')[1],"%H:%M:%S.%f") + timedelta(seconds=float(annotation_df.iloc[-1].time_interval.split(',')[2][:-1]))).time())
        result_dict['new_last'] = {
            'time_interval': [last_times+".000", '0'+str(timedelta(seconds=real_last_frame/fps))+'.000',annotation_df.iloc[-1].time_interval.split(',')[2][:-1]],
            'frame_interval':  [last_frames[1]+fps, real_last_frame, fps],
            'personalities': [],
        }
        
    return  pd.DataFrame.from_dict(result_dict, orient='index').reset_index(names="old_index")


def hypothesis_intervals_df(face_groups : List[FaceGroup], ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the pipeline results, generate a dataframe with the presence or absence of personalities for all the intervals in the reference dataframe
    """

    hyp_df = ref_df.copy().drop(columns=['personalities', 'old_index'])
    hyp_personalities = {i:set() for i in ref_df.index}
    hyp_in_frames = {i:[] for i in ref_df.index}
    
    for face_group in face_groups:
        person_id = face_group.person_id[0]
        
        for face in face_group.faces:
            frame_id = int(face.image_id)
            
            for index, row in ref_df.iterrows():
                frame_interval = list(range(row.frame_interval[0], row.frame_interval[1]+row.frame_interval[2],row.frame_interval[2]))

                # if the current detected face frame is within the interval
                # add the personality id to the hypothesis interval
                if frame_id*row.frame_interval[2] in frame_interval:

                    hyp_personalities[index].add(person_id)
                    hyp_in_frames[index].append((person_id,frame_id))
                    break
    
    hyp_df['personalities'] = hyp_personalities
    hyp_df['in_frames'] = hyp_in_frames
    
    return hyp_df


def video_eval(data_path:pathlib.Path, file_id:str, face_groups:List[FaceGroup]) -> pd.DataFrame:
    """
    Compute the video evaluation metrics from the reference annotation file and the pipeline results
    """
    vidObj = cv2.VideoCapture(str(data_path / "videos" / f'{file_id}.mp4'))
    nb_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(round(vidObj.get(cv2.CAP_PROP_FPS),0))
    
    with open(data_path / "annotations" / f'{file_id}.json', "r") as ref_file:
        ref_dict = json.load(ref_file)
    ref_df = generate_all_intervals(ref_dict, nb_frames, fps)
        
    hyp_df = hypothesis_intervals_df(face_groups, ref_df)

    all_ref_perso = {perso for personalities in ref_df.personalities for perso in personalities}
    all_hyp_perso = {perso for personalities in hyp_df.personalities for perso in personalities}
    classes_perso = list(all_ref_perso.union(all_hyp_perso))
    mlb = MultiLabelBinarizer(classes=classes_perso)
    
    ref_labels = mlb.fit_transform(ref_df.personalities)
    hyp_labels = mlb.fit_transform(hyp_df.personalities)

    return classification_report(ref_labels, hyp_labels, target_names=classes_perso,  zero_division=0, output_dict=True)