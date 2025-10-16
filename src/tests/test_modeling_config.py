import pandas as pd

from src.utils.modeling import FeatureConfig, compile_feature_list, one_hot_encode, build_design_matrix


def test_compile_feature_list_regex_and_exact():
    df = pd.DataFrame({
        'A': [1,2], 'B1': [3,4], 'B2': [5,6], 'TARGET': [0,1], 'OFFSET': [1.0, 2.0]
    })
    cfg = FeatureConfig(
        response='TARGET', offset='OFFSET',
        include=['A', '^B'], exclude=['B2'], categorical=[], min_cat_freq=0.01
    )
    feats = compile_feature_list(df, cfg)
    assert feats == ['A', 'B1']


def test_one_hot_and_design_matrix():
    df = pd.DataFrame({
        'TARGET': [1, 0, 2, 3],
        'OFFSET': [10.0, 20.0, 30.0, 40.0],
        'CAT': ['x', 'y', 'x', 'z'],
        'NUM': [1.0, 2.0, 3.0, 4.0],
    })
    cfg = FeatureConfig(
        response='TARGET', offset='OFFSET', include=['^CAT_', '^NUM$'], exclude=[], categorical=['CAT'], min_cat_freq=0.2
    )
    X, y, off = build_design_matrix(df, cfg)
    assert 'NUM' in X.columns
    # CAT should be expanded
    assert any(col.startswith('CAT_') for col in X.columns)
    assert len(y) == 4
    assert off is not None

