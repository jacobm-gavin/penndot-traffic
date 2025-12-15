import os
from typing import Optional

import numpy as np
import pandas as pd

from paths import DATA_DIR


def optionally_derive_alignment_rate() -> pd.DataFrame:
	veh_path = os.path.join(DATA_DIR, "VEHICLE_2024.csv")
	h3_path = os.path.join(DATA_DIR, "CRASH_2024_h3.csv")
	if not (os.path.exists(veh_path) and os.path.exists(h3_path)):
		return pd.DataFrame()
	try:
		veh = pd.read_csv(veh_path, usecols=["CRN", "RDWY_ALIGNMENT"], dtype=str)
		h3 = pd.read_csv(h3_path, usecols=["CRN", "H3_R8"], dtype={"CRN": str, "H3_R8": str})
	except Exception:
		return pd.DataFrame()
	veh["is_curve"] = veh["RDWY_ALIGNMENT"].str.contains("CURVE", case=False, na=False)
	crash_curve = veh.groupby("CRN")["is_curve"].max().reset_index()
	crash_curve = crash_curve.merge(h3, on="CRN", how="inner")
	h3_curve = crash_curve.groupby("H3_R8")["is_curve"].mean().reset_index()
	h3_curve.rename(columns={"is_curve": "RDWY_ALIGNMENT_CURVE_RATE"}, inplace=True)
	return h3_curve


def optionally_derive_surface_rates() -> pd.DataFrame:
	crash_path = os.path.join(DATA_DIR, "CRASH_2024.csv")
	h3_path = os.path.join(DATA_DIR, "CRASH_2024_h3.csv")
	if not (os.path.exists(crash_path) and os.path.exists(h3_path)):
		return pd.DataFrame()
	try:
		crash = pd.read_csv(crash_path, usecols=["CRN", "RDWY_SURF_TYPE_CD"], dtype=str)
		h3 = pd.read_csv(h3_path, usecols=["CRN", "H3_R8"], dtype={"CRN": str, "H3_R8": str})
	except Exception:
		return pd.DataFrame()
	crash = crash.merge(h3, on="CRN", how="inner")
	s = crash["RDWY_SURF_TYPE_CD"].str.upper().fillna("")
	cat = np.where(
		s.str.contains("ASPH", na=False),
		"ASPHALT",
		np.where(
			s.str.contains("CONC|PCC", na=False),
			"CONCRETE",
			np.where(s.str.contains("GRAV|STONE", na=False), "GRAVEL", "OTHER"),
		),
	)
	crash["SURFACE_CAT"] = cat
	counts = crash.groupby(["H3_R8", "SURFACE_CAT"]).size().reset_index(name="n")
	totals = counts.groupby("H3_R8")["n"].sum().reset_index(name="total")
	counts = counts.merge(totals, on="H3_R8", how="left")
	counts["rate"] = counts["n"] / counts["total"].replace(0, np.nan)
	pivot = counts.pivot(index="H3_R8", columns="SURFACE_CAT", values="rate").fillna(0.0)
	pivot = pivot.reset_index()
	pivot.columns = ["H3_R8"] + [f"RDWY_SURF_{c}_RATE" for c in pivot.columns[1:]]
	return pivot


def optionally_derive_illumination_rates() -> pd.DataFrame:
	crash_path = os.path.join(DATA_DIR, "CRASH_2024.csv")
	h3_path = os.path.join(DATA_DIR, "CRASH_2024_h3.csv")
	if not (os.path.exists(crash_path) and os.path.exists(h3_path)):
		return pd.DataFrame()
	try:
		crash = pd.read_csv(crash_path, usecols=["CRN", "ILLUMINATION"], dtype=str)
		h3 = pd.read_csv(h3_path, usecols=["CRN", "H3_R8"], dtype={"CRN": str, "H3_R8": str})
	except Exception:
		return pd.DataFrame()
	crash = crash.merge(h3, on="CRN", how="inner")
	illum = crash["ILLUMINATION"].str.upper().fillna("")
	cond_day = illum.str.contains("DAY", na=False)
	cond_dark = illum.str.contains("DARK", na=False)
	cond_lighted = illum.str.contains("LIGHT", na=False)
	crash["ILLUM_CAT"] = np.where(
		cond_day,
		"DAYLIGHT",
		np.where(cond_dark & cond_lighted, "DARK_LIGHTED", np.where(cond_dark & (~cond_lighted), "DARK_UNLIGHTED", "OTHER")),
	)
	counts = crash.groupby(["H3_R8", "ILLUM_CAT"]).size().reset_index(name="n")
	totals = counts.groupby("H3_R8")["n"].sum().reset_index(name="total")
	counts = counts.merge(totals, on="H3_R8", how="left")
	counts["rate"] = counts["n"] / counts["total"].replace(0, np.nan)
	pivot = counts.pivot(index="H3_R8", columns="ILLUM_CAT", values="rate").fillna(0.0)
	pivot = pivot.reset_index()
	pivot.columns = ["H3_R8"] + [f"ILLUM_{c}_RATE" for c in pivot.columns[1:]]
	return pivot


def add_optional_factors(h3_df: pd.DataFrame) -> pd.DataFrame:
	"""Merge derived alignment/surface/illumination rates when available."""
	if "RDWY_ALIGNMENT_CURVE_RATE" not in h3_df.columns:
		align_df = optionally_derive_alignment_rate()
		if not align_df.empty:
			h3_df = h3_df.merge(align_df, on="H3_R8", how="left")
	surface_df = optionally_derive_surface_rates()
	if not surface_df.empty:
		h3_df = h3_df.merge(surface_df, on="H3_R8", how="left")
	illum_df = optionally_derive_illumination_rates()
	if not illum_df.empty:
		h3_df = h3_df.merge(illum_df, on="H3_R8", how="left")
	return h3_df
