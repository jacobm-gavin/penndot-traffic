import os
from typing import Dict

import numpy as np
import pandas as pd

try:
	import statsmodels.api as sm
except Exception:
	sm = None

from paths import DATA_DIR
from data_sources import load_h3_aggregates, load_label_maps
from derivations import add_optional_factors


def derive_flag_rate_from_flags(flag_col: str) -> pd.DataFrame:
	"""Compute per-H3 rate for a given flag column in FLAGS_2024.csv."""
	flags_path = os.path.join(DATA_DIR, "FLAGS_2024.csv")
	h3_path = os.path.join(DATA_DIR, "CRASH_2024_h3.csv")
	if not (os.path.exists(flags_path) and os.path.exists(h3_path)):
		return pd.DataFrame()
	try:
		usecols = ["CRN", flag_col]
		flags = pd.read_csv(flags_path, usecols=usecols)
		h3 = pd.read_csv(h3_path, usecols=["CRN", "H3_R8"], dtype={"CRN": str, "H3_R8": str})
	except Exception:
		return pd.DataFrame()
	flags[flag_col] = pd.to_numeric(flags[flag_col], errors="coerce").fillna(0).astype(float)
	flags["CRN"] = flags["CRN"].astype(str)
	flags = flags.merge(h3, on="CRN", how="inner")
	grp = flags.groupby("H3_R8")[flag_col]
	rate = (grp.sum() / grp.count().replace(0, np.nan)).fillna(0).reset_index(name=f"FLAG_{flag_col}_RATE")
	return rate


def ensure_flag_factor(h3_df: pd.DataFrame, factor: str) -> pd.DataFrame:
	"""Ensure a FLAG_*_RATE factor exists by deriving from FLAGS_2024 if necessary."""
	if factor in h3_df.columns:
		return h3_df
	if factor.startswith("FLAG_"):
		raw = factor[len("FLAG_"):]
		if raw.endswith("_RATE"):
			raw = raw[:-5]
		flag_df = derive_flag_rate_from_flags(raw)
		if not flag_df.empty and factor in flag_df.columns:
			h3_df = h3_df.merge(flag_df, on="H3_R8", how="left")
	return h3_df


def analyze_flag_illumination_dark() -> None:
	"""Assess predictiveness of FLAGS_2024.ILLUMINATION_DARK for crash counts per H3."""
	h3 = load_h3_aggregates()
	flag_df = derive_flag_rate_from_flags("ILLUMINATION_DARK")
	if flag_df.empty:
		raise RuntimeError("Could not derive FLAG_ILLUMINATION_DARK_RATE from FLAGS_2024.csv")
	h3 = h3.merge(flag_df, on="H3_R8", how="left")
	fcol = "FLAG_ILLUMINATION_DARK_RATE"
	from plotting import plot_factor_relationships  # Delayed import to avoid cycle

	plot_factor_relationships(h3, [fcol], load_label_maps())
	if sm is None:
		print("statsmodels not available; skipping GLM.")
		return
	X = h3[[fcol]].copy()
	X = sm.add_constant(X)
	y = h3["CRASH_COUNT"].astype(float)
	offset = h3["LOG_AADT"].astype(float)
	mask_X = np.asarray(~pd.isna(X).any(axis=1))
	mask_y = np.asarray(~pd.isna(y))
	mask_off = np.asarray(~pd.isna(offset))
	valid = mask_X & mask_y & mask_off
	X, y, offset = X[valid], y[valid], offset[valid]
	if len(X) < 50:
		print("Insufficient rows for GLM; skipping.")
		return
	try:
		model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
		res = model.fit()
	except Exception:
		model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
		res = model.fit()
	params = res.params.drop("const", errors="ignore")
	conf = res.conf_int().loc[params.index]
	irr = np.exp(params)
	irr_lo = np.exp(conf[0])
	irr_hi = np.exp(conf[1])
	print("ILLUMINATION_DARK flag predictiveness (IRR):")
	for name, val in irr.items():
		print(f" - {name}: IRR={val:.3f} (CI {float(irr_lo.loc[name]):.3f}–{float(irr_hi.loc[name]):.3f})")
