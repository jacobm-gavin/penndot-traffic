import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from utils.severity import derive_serious_from_max_severity
import pandas.errors as pd_errors


CRASH_COLS_KEEP: List[str] = [
    "CRN",
    "H3_R8",
    "MAX_SEVERITY_LEVEL",
    # Selected context columns to aggregate/share
    "URBAN_RURAL",
    "INTERSECTION_RELATED",
    "WORK_ZONE_IND",
    "WORK_ZONE_TYPE",
    "WORK_ZONE_LOC",
    "ILLUMINATION",
    "ROAD_CONDITION",
    "RDWY_SURF_TYPE_CD",
    "WEATHER1",
    "WEATHER2",
    # counts
    "VEHICLE_COUNT",
    "HEAVY_TRUCK_COUNT",
]


def read_csv_upper(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def build_features(
    crash_csv: str,
    crash_h3_csv: str,
    exposure_csv: str,
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    crash_base = read_csv_upper(crash_csv)
    crash_h3 = read_csv_upper(crash_h3_csv)
    # Read exposure; tolerate empty file by creating a minimal empty frame
    try:
        exposure_df = read_csv_upper(exposure_csv)
    except pd_errors.EmptyDataError:
        exposure_df = pd.DataFrame(columns=["H3_R8", "EXPOSURE_AADT_MXM_365"])

    # Ensure required columns
    for col in ["CRN"]:
        if col not in crash_base.columns:
            raise ValueError(f"Missing {col} in crash data {crash_csv}")
    for col in ["CRN", "H3_R8"]:
        if col not in crash_h3.columns:
            raise ValueError(f"Missing {col} in crash H3 data {crash_h3_csv}")

    # Join H3 onto full crash records
    crash_df = crash_base.merge(crash_h3[["CRN", "H3_R8"]], on="CRN", how="inner")

    # Derive SERIOUS
    crash_df["SERIOUS"] = crash_df["MAX_SEVERITY_LEVEL"].apply(derive_serious_from_max_severity)

    # Aggregate per H3 cell
    group_cols = ["H3_R8"]
    agg_specs = {
        "CRN": "count",
        "SERIOUS": "sum",
        "VEHICLE_COUNT": "mean",
        "HEAVY_TRUCK_COUNT": "mean",
    }
    base = crash_df.groupby(group_cols).agg(agg_specs).rename(columns={
        "CRN": "CRASH_COUNT_CELL",
        "SERIOUS": "SERIOUS_COUNT_CELL",
    })

    # Shares for binary indicators
    def mean_share(col: str):
        if col in crash_df.columns:
            base[f"{col}_SHARE"] = crash_df.groupby("H3_R8")[col].apply(
                lambda s: pd.to_numeric(s, errors="coerce").fillna(0).astype(float).mean()
            )

    for bin_col in ["INTERSECTION_RELATED", "WORK_ZONE_IND"]:
        mean_share(bin_col)

    # Proportions for categorical fields: add one-hot then mean
    def add_cat_shares(col: str, top_k: int = 6):
        if col not in crash_df.columns:
            return
        vc = crash_df[col].astype(str).fillna("NA").value_counts()
        cats = list(vc.index[:top_k])
        for v in cats:
            mask = (crash_df[col].astype(str).fillna("NA") == v).astype(int)
            base[f"{col}_{v}_SHARE"] = mask.groupby(crash_df["H3_R8"]).mean()

    for cat in [
        "ILLUMINATION",
        "ROAD_CONDITION",
        "RDWY_SURF_TYPE_CD",
        "WEATHER1",
        "WEATHER2",
        "WORK_ZONE_TYPE",
        "WORK_ZONE_LOC",
        "URBAN_RURAL",
    ]:
        add_cat_shares(cat)

    # Join exposure
    if "H3_R8" not in exposure_df.columns:
        # If exposure is empty, create the column to allow merge
        exposure_df["H3_R8"] = []
    features = base.reset_index().merge(
        exposure_df, on="H3_R8", how="left"
    )
    # Replace missing exposure with small positive to avoid log(0)
    # Ensure exposure column exists and is positive
    if "EXPOSURE_AADT_MXM_365" not in features.columns:
        features["EXPOSURE_AADT_MXM_365"] = 1.0
    else:
        features["EXPOSURE_AADT_MXM_365"] = features["EXPOSURE_AADT_MXM_365"].fillna(1.0).clip(lower=1e-6)

    features.to_csv(out_csv, index=False)
    print(f"Wrote features to {out_csv} with {len(features)} H3 cells")


def main():
    parser = argparse.ArgumentParser(description="Build H3 cell features by aggregating crash-level data and joining exposure")
    parser.add_argument("--crash", default="data/CRASH_2024.csv", help="Crash CSV (full)")
    parser.add_argument("--crash-h3", default="data/CRASH_2024_h3.csv", help="Crash CSV with CRN and H3_R8 column")
    parser.add_argument("--exposure", default="data/H3_res8_exposure.csv", help="Exposure per H3 cell CSV")
    parser.add_argument("--output", default="data/aggregates/h3_features.csv", help="Output features CSV")
    args = parser.parse_args()

    build_features(args.crash, args.crash_h3, args.exposure, args.output)


if __name__ == "__main__":
    main()
