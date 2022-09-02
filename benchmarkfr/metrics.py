from pyannote.metrics.identification import IdentificationPrecision, IdentificationRecall, IdentificationErrorRate
from pyannote.core import Annotation, Segment
import pandas as pd


def eval_interval(reference: pd.DataFrame, hypothesis: pd.DataFrame):
    hypothesis_pya = Annotation()
    reference_pya = Annotation()

    for idx, row in hypothesis.iterrows():
        hypothesis_pya[Segment(row["start"], row["end"] + 1), idx] = row["name"]

    for idx, row in reference.iterrows():
        reference_pya[Segment(row["start"], row["end"] + 1), idx] = row["name"]

    return IdentificationPrecision()(reference_pya, hypothesis_pya), IdentificationRecall()(reference_pya,

    # TO DO :
    # metric per personality
    # metric per interval
