import json
import pandas as pd
import pytest

from benchmarkfr.metrics import eval_interval


def test_eval_interval():
    df_ref = pd.DataFrame({"start": [1], "end": [1], "name": ["John Smith"]})
    df_hyp = pd.DataFrame({"start": [1], "end": [1], "name": ["John Smith"]})

    assert eval_interval(df_ref, df_hyp) == (1, 1)


def test_eval_interval_2():
    df_ref = pd.DataFrame({"start": [1], "end": [1], "name": ["John Smith"]})
    df_hyp = pd.DataFrame({"start": [1], "end": [1], "name": ["John Do"]})

    assert eval_interval(df_ref, df_hyp) == (0, 0)


def test_eval_interval_3():
    df_ref = pd.DataFrame({"start": [1], "end": [1], "name": ["John Smith"]})
    df_hyp = pd.DataFrame({"start": [1], "end": [2], "name": ["John Smith"]})

    assert eval_interval(df_ref, df_hyp) == (0.5, 1)


def test_eval_interval_4():
    df_ref = pd.DataFrame({"start": [1, 1], "end": [1, 1], "name": ["John Smith", "Claudia Smith"]})
    df_hyp = pd.DataFrame({"start": [1], "end": [1], "name": ["John Smith"]})

    assert eval_interval(df_ref, df_hyp) == (1, 0.5)


def test_eval_interval_5():
    df_ref = pd.DataFrame({"start": [1, 1], "end": [1, 1], "name": ["John Smith", "Claudia Smith"]})
    df_hyp = pd.DataFrame({"start": [1], "end": [1], "name": ["John Smith"]})

    assert eval_interval(df_ref, df_hyp) == (1, 0.5)


@pytest.mark.skip
def test_eval_interval_6():
    df_hyp = pd.DataFrame({"start": [1, 7], "end": [10, 10], "name": ["John Smith", "Claudia Smith"]})
    df_ref = pd.DataFrame({"start": [1], "end": [10], "name": ["John Smith"]})

    assert eval_interval(df_ref, df_hyp) == (1, 0.5)


def test_eval_interval_7():
    with open("data/galleries/vivement-dimanche-6.json") as f:
        expected_result = json.load(f)
        # "annotation": {"0": {"time_interval": "[00:00:00.000,00:00:22.000,1.0]", "frame_interval": "[0,550,25]", "personalities": ["Olivier de Kersauson", "Michel Drucker"]},
        d_ref = {"start": [], "end": [], "name": []}
        for annotation in expected_result["annotation"].values():
            d_ref["start"].extend(
                [int(json.loads(annotation["frame_interval"])[0])] * len(annotation["personalities"]))
            d_ref["end"].extend(
                [int(json.loads(annotation["frame_interval"])[1])] * len(annotation["personalities"]))
            d_ref["name"].extend(annotation["personalities"])
        df_ref = pd.DataFrame(d_ref)

    with open("vivement-dimanche-6.json") as f:
        result = json.load(f)
        # "annotation": {"0": {"time_interval": "[00:00:00.000,00:00:22.000,1.0]", "frame_interval": "[0,550,25]", "personalities": ["Olivier de Kersauson", "Michel Drucker"]},
        d_hyp = {"start": [], "end": [], "name": []}
        for annotation in expected_result["annotation"].values():
            start= json.loads(annotation["frame_interval"])[0]
            end = json.loads(annotation["frame_interval"])[1]
            for face_group in result:
                for face in face_group["faces"]:
                    if start / 25 <= int(face["image_id"]) <= end / 25:
                        d_hyp["start"].append(start)
                        d_hyp["end"].append(end)
                        d_hyp["name"].append(face_group["person_id"][0])
                        continue
        df_hyp = pd.DataFrame(d_hyp)

    assert eval_interval(df_ref, df_hyp) == (1, 1)
