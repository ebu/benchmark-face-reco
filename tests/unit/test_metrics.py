from benchmarkfr.metrics import eval_interval
import pandas as pd

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

    df_ref = pd.DataFrame({"start": [1,1], "end": [1,1], "name": ["John Smith","Claudia Smith"]})
    df_hyp = pd.DataFrame({"start": [1], "end": [1], "name": ["John Smith"]})

    assert eval_interval(df_ref, df_hyp) == (1, 0.5)

def test_eval_interval_5():

    df_ref = pd.DataFrame({"start": [1,1], "end": [1,1], "name": ["John Smith","Claudia Smith"]})
    df_hyp = pd.DataFrame({"start": [1], "end": [1], "name": ["John Smith"]})

    assert eval_interval(df_ref, df_hyp) == (1, 0.5)

def test_eval_interval_5():
    df_hyp = pd.DataFrame({"start": [1, 7], "end": [10, 10], "name": ["John Smith", "Claudia Smith"]})
    df_ref = pd.DataFrame({"start": [1], "end": [10], "name": ["John Smith"]})

    assert eval_interval(df_ref, df_hyp) == (1, 0.5)