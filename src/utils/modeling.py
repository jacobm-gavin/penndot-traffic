from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class FeatureConfig:
    response: str
    offset: Optional[str]
    include: List[str]
    exclude: List[str]
    categorical: List[str]
    min_cat_freq: float = 0.01


def compile_feature_list(df: pd.DataFrame, cfg: FeatureConfig) -> List[str]:
    cols = set()
    for pat in cfg.include:
        if pat in df.columns:
            cols.add(pat)
            continue
        # regex inclusion
        regex = re.compile(pat)
        for c in df.columns:
            if regex.search(c):
                cols.add(c)
    for pat in cfg.exclude:
        if pat in cols:
            cols.remove(pat)
        else:
            # drop matching by regex
            regex = re.compile(pat)
            cols = {c for c in cols if not regex.search(c)}

    # Never include response or offset in predictors
    cols.discard(cfg.response)
    if cfg.offset:
        cols.discard(cfg.offset)
    # Fallback: if no features matched, select all numeric columns except response/offset and obvious IDs
    if not cols:
        for c in df.columns:
            if c in (cfg.response, cfg.offset):
                continue
            if c.upper().startswith("H3_"):
                continue
            if is_numeric_dtype(df[c]):
                # Respect exclude patterns
                excluded = False
                for pat in cfg.exclude:
                    regex = re.compile(pat)
                    if (pat == c) or regex.search(c):
                        excluded = True
                        break
                if not excluded:
                    cols.add(c)
    return sorted(cols)


def one_hot_encode(
    df: pd.DataFrame,
    categorical_cols: List[str],
    min_freq: float = 0.01,
) -> pd.DataFrame:
    """
    One-hot encode categorical columns with frequency threshold.
    Keeps categories with proportion >= min_freq; others grouped into 'OTHER'.
    """
    out = df.copy()
    for col in categorical_cols:
        if col not in out.columns:
            continue
        value_counts = out[col].astype(str).fillna("NA").value_counts(normalize=True)
        keep = set(value_counts[value_counts >= min_freq].index)
        series = out[col].astype(str).fillna("NA").apply(lambda v: v if v in keep else "OTHER")
        dummies = pd.get_dummies(series, prefix=col)
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
    return out


def build_design_matrix(df: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Build X, y, offset from df according to config, applying one-hot encoding.
    """
    # Ensure uppercase columns for safety
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Response and offset
    y = pd.to_numeric(df[cfg.response], errors="coerce")
    offset = pd.to_numeric(df[cfg.offset], errors="coerce") if cfg.offset and cfg.offset in df.columns else None

    # Categorical encoding
    df_enc = one_hot_encode(df, cfg.categorical, min_freq=cfg.min_cat_freq)

    features = compile_feature_list(df_enc, cfg)
    # Force numeric and fill missing with 0 for robustness
    X = df_enc[features].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.fillna(0.0)

    return X, y, offset
