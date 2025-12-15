import os
from typing import Dict

import pandas as pd

from paths import AGG_DIR


def load_h3_aggregates() -> pd.DataFrame:
	"""Load the core H3 modeling dataset with sanity checks."""
	path = os.path.join(AGG_DIR, "h3_modeling_dataset.csv")
	df = pd.read_csv(path)
	expected = ["H3_R8", "CRASH_RATE", "CRASH_COUNT", "LOG_AADT"]
	missing = [c for c in expected if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in h3_modeling_dataset: {missing}")
	return df


def load_label_maps() -> Dict[str, str]:
	"""Collect feature/label mappings from optional helper CSVs."""
	label_map: Dict[str, str] = {}
	fmap_path = os.path.join(AGG_DIR, "gbm_feature_map.csv")
	if os.path.exists(fmap_path):
		try:
			fmap = pd.read_csv(fmap_path)
			name_col = "name" if "name" in fmap.columns else None
			feat_col = "feature" if "feature" in fmap.columns else None
			if name_col and feat_col:
				for _, r in fmap.iterrows():
					label_map[str(r[feat_col])] = str(r[name_col])
		except Exception:
			pass
	dtype_map_path = os.path.join(AGG_DIR, "predictor_dtype_mapping.csv")
	if os.path.exists(dtype_map_path):
		try:
			dmap = pd.read_csv(dtype_map_path)
			if {"predictor", "readable_name"}.issubset(set(dmap.columns)):
				for _, r in dmap.iterrows():
					label_map[str(r["predictor"])] = str(r["readable_name"])
		except Exception:
			pass
	return label_map
