import os
import sys
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

try:
	import statsmodels.api as sm
except Exception:
	sm = None


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AGG_DIR = os.path.join(DATA_DIR, "aggregates")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


ROAD_FACTOR_COLUMNS = [
	"SPEED_LIMIT_MEAN",
	"LANE_COUNT_MEAN",
	"WET_ROAD_RATE",
	"ICY_ROAD_RATE",
	"INTERSECTION_RATE",
	"WORK_ZONE_RATE",
	"STATE_ROAD_RATE",
	"TURNPIKE_RATE",
]


def ensure_plots_dir() -> None:
	os.makedirs(PLOTS_DIR, exist_ok=True)


def load_h3_aggregates() -> pd.DataFrame:
	path = os.path.join(AGG_DIR, "h3_modeling_dataset.csv")
	df = pd.read_csv(path)
	# Basic expected columns
	expected = ["H3_R8", "CRASH_RATE", "CRASH_COUNT", "LOG_AADT"]
	missing = [c for c in expected if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in h3_modeling_dataset: {missing}")
	return df


def load_label_maps() -> Dict[str, str]:
	label_map: Dict[str, str] = {}
	# Feature map (optional)
	fmap_path = os.path.join(AGG_DIR, "gbm_feature_map.csv")
	if os.path.exists(fmap_path):
		try:
			fmap = pd.read_csv(fmap_path)
			# Expect columns: feature, name (or similar). Fallback to identity if not present.
			name_col = "name" if "name" in fmap.columns else None
			feat_col = "feature" if "feature" in fmap.columns else None
			if name_col and feat_col:
				for _, r in fmap.iterrows():
					label_map[str(r[feat_col])] = str(r[name_col])
		except Exception:
			pass
	# Predictor dtype mapping may include readable names
	dtype_map_path = os.path.join(AGG_DIR, "predictor_dtype_mapping.csv")
	if os.path.exists(dtype_map_path):
		try:
			dmap = pd.read_csv(dtype_map_path)
			# Try common columns
			if {"predictor", "readable_name"}.issubset(set(dmap.columns)):
				for _, r in dmap.iterrows():
					label_map[str(r["predictor"])] = str(r["readable_name"])
		except Exception:
			pass
	return label_map


def readable_label(col: str, label_map: Dict[str, str]) -> str:
	return label_map.get(col, col.replace("_", " ").title())


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

	# Define curved vs straight from RDWY_ALIGNMENT codes
	veh["is_curve"] = veh["RDWY_ALIGNMENT"].str.contains("CURVE", case=False, na=False)
	# Aggregate to crash level (any curve among units => curve)
	crash_curve = veh.groupby("CRN")["is_curve"].max().reset_index()
	crash_curve = crash_curve.merge(h3, on="CRN", how="inner")
	# Aggregate to H3: proportion of curve crashes
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
	# Simplify surface types to categories; treat codes by prefix where possible
	# We will create rates for Asphalt, Concrete, Gravel, Other
	s = crash["RDWY_SURF_TYPE_CD"].str.upper().fillna("")
	cat = np.where(s.str.contains("ASPH", na=False), "ASPHALT",
		  np.where(s.str.contains("CONC|PCC", na=False), "CONCRETE",
		  np.where(s.str.contains("GRAV|STONE", na=False), "GRAVEL",
				   "OTHER")))
	crash["SURFACE_CAT"] = cat

	# Compute per H3 category rates
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
	# Normalize common categories
	# Vectorized category mapping
	cond_day = illum.str.contains("DAY", na=False)
	cond_dark = illum.str.contains("DARK", na=False)
	cond_lighted = illum.str.contains("LIGHT", na=False)
	crash["ILLUM_CAT"] = np.where(
		cond_day,
		"DAYLIGHT",
		np.where(
			cond_dark & cond_lighted,
			"DARK_LIGHTED",
			np.where(cond_dark & (~cond_lighted), "DARK_UNLIGHTED", "OTHER")
		)
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
	# Alignment
	if "RDWY_ALIGNMENT_CURVE_RATE" not in h3_df.columns:
		align_df = optionally_derive_alignment_rate()
		if not align_df.empty:
			h3_df = h3_df.merge(align_df, on="H3_R8", how="left")
	# Surface
	surface_df = optionally_derive_surface_rates()
	if not surface_df.empty:
		h3_df = h3_df.merge(surface_df, on="H3_R8", how="left")
	# Illumination
	illum_df = optionally_derive_illumination_rates()
	if not illum_df.empty:
		h3_df = h3_df.merge(illum_df, on="H3_R8", how="left")
	return h3_df


def plot_factor_relationships(h3_df: pd.DataFrame, factors: List[str], label_map: Dict[str, str]) -> None:
	ensure_plots_dir()
	sns.set(style="whitegrid")
	for col in factors:
		if col not in h3_df.columns:
			continue
		x = h3_df[col]
		y = h3_df["CRASH_RATE"]
		# Drop NaNs
		valid = (~x.isna()) & (~y.isna())
		x = x[valid]
		y = y[valid]
		if len(x) < 10:
			continue
		fig, ax = plt.subplots(figsize=(7, 5))
		ax.scatter(x, y, s=12, alpha=0.35, color="#1f77b4")
		# Binned means
		try:
			q = pd.qcut(x, q=10, duplicates='drop')
			binned = pd.DataFrame({"x": x.groupby(q).mean(), "y": y.groupby(q).mean()})
			ax.plot(binned["x"], binned["y"], color="#d62728", linewidth=2, label="Binned mean")
		except Exception:
			pass
		ax.set_xlabel(readable_label(col, label_map))
		ax.set_ylabel("Crash Rate (per exposure)")
		ax.set_title(f"Crash Rate vs {readable_label(col, label_map)}")
		ax.legend(loc="best")
		out = os.path.join(PLOTS_DIR, f"road_factor_{col}.png")
		plt.tight_layout()
		fig.savefig(out, dpi=150)
		plt.close(fig)


def fit_glm_and_plot(h3_df: pd.DataFrame, factors: List[str]) -> None:
	if sm is None:
		print("statsmodels not available; skipping GLM.")
		return
	ensure_plots_dir()
	# Prepare design matrix
	cols = [c for c in factors if c in h3_df.columns]
	X = h3_df[cols].copy()
	y = h3_df["CRASH_COUNT"].astype(float)
	offset = h3_df["LOG_AADT"].astype(float)
	# Drop rows with NaNs
	valid = (~pd.isna(X).any(axis=1)) & (~pd.isna(y)) & (~pd.isna(offset))
	X = X[valid]
	y = y[valid]
	offset = offset[valid]
	if len(X) < 50:
		print("Insufficient rows for GLM; skipping.")
		return
	X = sm.add_constant(X)
	try:
		model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
		res = model.fit()
	except Exception:
		try:
			model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
			res = model.fit()
		except Exception as e:
			print(f"GLM failed: {e}")
			return

	params = res.params.drop("const", errors="ignore")
	conf = res.conf_int().loc[params.index]
	irr = np.exp(params)
	irr_lo = np.exp(conf[0])
	irr_hi = np.exp(conf[1])
	df_plot = pd.DataFrame({
		"factor": params.index,
		"IRR": irr,
		"IRR_lo": irr_lo,
		"IRR_hi": irr_hi,
	}).sort_values("IRR", ascending=False)

	fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df_plot))))
	ax.barh(df_plot["factor"], df_plot["IRR"], xerr=[df_plot["IRR"] - df_plot["IRR_lo"], df_plot["IRR_hi"] - df_plot["IRR"]], color="#2ca02c", alpha=0.8)
	ax.invert_yaxis()
	ax.set_xlabel("Incidence Rate Ratio (IRR)")
	ax.set_title("GLM Road Factor Effects on Crash Count (offset by AADT)")
	plt.tight_layout()
	out = os.path.join(PLOTS_DIR, "irr_ranked.png")
	fig.savefig(out, dpi=150)
	plt.close(fig)


def predict_glm_counts(h3_df: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
	"""Fit a Poisson GLM with offset(LOG_AADT) on the given factors and
	return a DataFrame with predictions per H3 tile: PRED_CRASH_COUNT and PRED_CRASH_RATE.
	"""
	if sm is None:
		raise RuntimeError("statsmodels not available; cannot predict GLM.")
	cols = [c for c in factors if c in h3_df.columns]
	if not cols:
		raise ValueError("No valid factor columns found for prediction.")
	X = h3_df[cols].copy()
	y = h3_df["CRASH_COUNT"].astype(float)
	offset = h3_df["LOG_AADT"].astype(float)
	valid = (~pd.isna(X).any(axis=1)) & (~pd.isna(y)) & (~pd.isna(offset))
	X = X[valid]
	y = y[valid]
	offset = offset[valid]
	idx = h3_df.index[valid]
	X_const = sm.add_constant(X)
	try:
		model = sm.GLM(y, X_const, family=sm.families.Poisson(), offset=offset)
		res = model.fit()
	except Exception:
		model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial(), offset=offset)
		res = model.fit()
	# Linear predictor: eta = const + X*beta + offset
	eta = res.predict(X_const, offset=offset, linear=True)
	mu = np.exp(eta)
	# Rate = mu / AADT = exp(eta - offset)
	rate = np.exp(eta - offset)
	pred_df = pd.DataFrame({
		"H3_R8": h3_df.loc[idx, "H3_R8"].values,
		"PRED_CRASH_COUNT": mu,
		"PRED_CRASH_RATE": rate,
	})
	return pred_df


def evaluate_glm(h3_df: pd.DataFrame, factors: List[str], test_frac: float = 0.2, use_spatial_blocks: bool = True) -> Dict[str, float]:
	"""Evaluate GLM with a train/test split.
	- If spatial blocks file exists and use_spatial_blocks=True, split by blocks to reduce leakage.
	- Returns metrics: MAE, RMSE, PoissonDeviance.
	"""
	if sm is None:
		raise RuntimeError("statsmodels not available; cannot evaluate GLM.")
	cols = [c for c in factors if c in h3_df.columns]
	if not cols:
		raise ValueError("No valid factor columns found for evaluation.")
	# Build dataset
	df = h3_df[["H3_R8", "CRASH_COUNT", "LOG_AADT"] + cols].dropna()
	if df.empty:
		raise ValueError("No rows available for evaluation after dropping NaNs.")
	# Spatial split if possible
	train_idx = None
	test_idx = None
	blocks_path = os.path.join(AGG_DIR, "spatial_blocks.csv")
	if use_spatial_blocks and os.path.exists(blocks_path):
		try:
			blocks = pd.read_csv(blocks_path)
			if {"H3_R8", "block_id"}.issubset(set(blocks.columns)):
				dfb = df.merge(blocks[["H3_R8", "block_id"]], on="H3_R8", how="left")
				# Pick random 20% of blocks for test
				unique_blocks = dfb["block_id"].dropna().unique()
				if len(unique_blocks) > 0:
					np.random.seed(42)
					test_blocks = set(np.random.choice(unique_blocks, size=max(1, int(len(unique_blocks) * test_frac)), replace=False))
					train_idx = dfb.index[~dfb["block_id"].isin(test_blocks)]
					test_idx = dfb.index[dfb["block_id"].isin(test_blocks)]
		except Exception:
			train_idx = None
			test_idx = None
	# Fallback random split
	if train_idx is None or test_idx is None or len(test_idx) == 0:
		np.random.seed(42)
		perm = np.random.permutation(len(df))
		n_test = max(1, int(len(df) * test_frac))
		test_idx = perm[:n_test]
		train_idx = perm[n_test:]
	# Train model
	train = df.iloc[train_idx]
	test = df.iloc[test_idx]
	X_train = sm.add_constant(train[cols])
	y_train = train["CRASH_COUNT"].astype(float)
	off_train = train["LOG_AADT"].astype(float)
	try:
		model = sm.GLM(y_train, X_train, family=sm.families.Poisson(), offset=off_train)
		res = model.fit()
	except Exception:
		model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(), offset=off_train)
		res = model.fit()
	# Predict on test
	X_test = sm.add_constant(test[cols])
	off_test = test["LOG_AADT"].astype(float)
	eta_test = res.predict(X_test, offset=off_test, linear=True)
	mu_test = np.exp(eta_test)
	y_true = np.asarray(test["CRASH_COUNT"].astype(float).values)
	# Metrics
	mae = float(np.mean(np.abs(y_true - mu_test)))
	rmse = float(np.sqrt(np.mean((y_true - mu_test) ** 2)))
	# Poisson deviance: 2 * sum( y*log(y/mu) - (y - mu) ), define y*log(y/mu)=0 when y=0
	with np.errstate(divide='ignore', invalid='ignore'):
		term = np.where(y_true > 0.0, y_true * np.log(y_true / np.asarray(mu_test)), 0.0)
		dev = 2.0 * np.sum(term - (y_true - np.asarray(mu_test)))
	return {"MAE": mae, "RMSE": rmse, "PoissonDeviance": float(dev)}


def plot_h3_choropleth(df: pd.DataFrame, factor: str, quantiles: Optional[int] = None, minimal: bool = False) -> None:
	"""Plot H3 polygons colored by factor value.
	If `quantiles` is provided (e.g., 10), values are quantile-scaled to decile ranks.
	Requires the `h3` Python package.
	"""
	ensure_plots_dir()
	try:
		import h3
	except Exception:
		raise RuntimeError("Please install the 'h3' package: pip install h3")

	patches = []
	colors = []
	vals = df[factor].astype(float)
	if quantiles and quantiles > 1:
		try:
			q = pd.qcut(vals, q=quantiles, labels=False, duplicates='drop')
			vals_scaled = q.astype(float)
			vmin, vmax = 0.0, float(vals_scaled.max())
			color_values = vals_scaled
			colorbar_label = f"{factor} (quantile rank)"
		except Exception:
			# Fallback to raw if qcut fails
			vmin, vmax = float(vals.min()), float(vals.max())
			color_values = vals
			colorbar_label = factor
	else:
		vmin, vmax = float(vals.min()), float(vals.max())
		color_values = vals
		colorbar_label = factor

	for i in range(len(df)):
		row = df.iloc[i]
		h = row["H3_R8"]
		val = float(row[factor])
		# Get boundary (lat, lng). Support both legacy and new API.
		boundary = h3.cell_to_boundary(h)
		# Convert to (lng, lat) for matplotlib
		poly = MplPolygon([(lng, lat) for lat, lng in boundary], closed=True)
		patches.append(poly)
		# Use scaled values if quantiles selected
		if quantiles and quantiles > 1 and 'vals_scaled' in locals():
			colors.append(float(vals_scaled.iloc[i]))
		else:
			colors.append(val)

	fig, ax = plt.subplots(figsize=(8, 8))
	pc = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidths=0.2)
	pc.set_array(np.array(colors))
	pc.set_clim(vmin=vmin, vmax=vmax)
	ax.add_collection(pc)
	ax.set_aspect("equal", adjustable="box")
	if minimal:
		ax.axis('off')
	else:
		ax.set_xlabel("Longitude")
		ax.set_ylabel("Latitude")
		ax.set_title(f"H3 Choropleth: {factor}")
		fig.colorbar(pc, ax=ax, label=colorbar_label)
	# Auto bounds
	all_points = np.vstack([np.array(p.get_xy()) for p in patches])
	ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
	ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
	plt.tight_layout()
	out = os.path.join(PLOTS_DIR, f"map_{factor}.png")
	fig.savefig(out, dpi=150)
	plt.close(fig)


def derive_flag_rate_from_flags(flag_col: str) -> pd.DataFrame:
	"""Compute per-H3 rate for a given flag column in FLAGS_2024.csv.
	Returns columns: H3_R8, FLAG_<flag_col>_RATE
	"""
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
	# Coerce to numeric 0/1
	flags[flag_col] = pd.to_numeric(flags[flag_col], errors="coerce").fillna(0).astype(float)
	# Ensure CRN as str for join
	flags["CRN"] = flags["CRN"].astype(str)
	flags = flags.merge(h3, on="CRN", how="inner")
	grp = flags.groupby("H3_R8")[flag_col]
	rate = (grp.sum() / grp.count().replace(0, np.nan)).fillna(0).reset_index(name=f"FLAG_{flag_col}_RATE")
	return rate


def analyze_flag_illumination_dark() -> None:
	"""Assess predictiveness of FLAGS_2024.ILLUMINATION_DARK for crash counts per H3.
	- Derives FLAG_ILLUMINATION_DARK_RATE
	- Fits Poisson/NB GLM: CRASH_COUNT ~ rate + offset(LOG_AADT)
	- Saves scatter and prints IRR with 95% CI
	"""
	h3 = load_h3_aggregates()
	flag_df = derive_flag_rate_from_flags("ILLUMINATION_DARK")
	if flag_df.empty:
		raise RuntimeError("Could not derive FLAG_ILLUMINATION_DARK_RATE from FLAGS_2024.csv")
	h3 = h3.merge(flag_df, on="H3_R8", how="left")
	fcol = "FLAG_ILLUMINATION_DARK_RATE"
	# Quick scatter plot vs crash rate
	plot_factor_relationships(h3, [fcol], load_label_maps())
	# GLM
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


def ensure_flag_factor(h3_df: pd.DataFrame, factor: str) -> pd.DataFrame:
	"""Ensure a FLAG_*_RATE factor exists by deriving from FLAGS_2024 if necessary.
	Returns potentially enriched h3_df.
	"""
	if factor in h3_df.columns:
		return h3_df
	if factor.startswith("FLAG_"):
		# Strip prefix and optional _RATE suffix to get raw flag column
		raw = factor[len("FLAG_"):]
		if raw.endswith("_RATE"):
			raw = raw[:-5]
		flag_df = derive_flag_rate_from_flags(raw)
		if not flag_df.empty and factor in flag_df.columns:
			h3_df = h3_df.merge(flag_df, on="H3_R8", how="left")
	return h3_df


def _detect_lat_lon_columns(df: pd.DataFrame) -> Optional[tuple[str, str]]:
	"""Try to detect latitude/longitude columns in a crash-level DataFrame.
	Returns (lat_col, lon_col) or None if not found.
	"""
	cols = {c.lower(): c for c in df.columns}
	# Common patterns
	lat_candidates = [
		"latitude", "lat", "dec_lat", "y", "gps_lat", "crash_latitude",
	]
	lon_candidates = [
		"longitude", "lon", "lng", "dec_long", "x", "gps_long", "crash_longitude",
	]
	lat_col = next((cols[c] for c in lat_candidates if c in cols), None)
	lon_col = next((cols[c] for c in lon_candidates if c in cols), None)
	if lat_col and lon_col:
		return lat_col, lon_col
	# Heuristic: first column containing 'lat' and first containing 'lon'/'lng'
	lat_guess = next((c for c in df.columns if "lat" in c.lower()), None)
	lon_guess = next((c for c in df.columns if ("lon" in c.lower() or "lng" in c.lower() or "long" in c.lower())), None)
	if lat_guess and lon_guess:
		return lat_guess, lon_guess
	return None


def plot_crash_dots_threshold(
	h3_df: pd.DataFrame,
	factor: str,
	threshold: float,
	from_h3: bool = True,
	color: str = "red",
	size: float = 6.0,
	minimal: bool = False,
) -> None:
	"""Plot crash-level dots for crashes where a factor exceeds a threshold.
	- If from_h3=True, the factor is taken from the H3 dataset and merged via CRASH_2024_h3.
	- If from_h3=False, the factor must exist in CRASH_2024.csv and will be filtered per crash.
	"""
	ensure_plots_dir()
	crash_path = os.path.join(DATA_DIR, "CRASH_2024.csv")
	h3_map_path = os.path.join(DATA_DIR, "CRASH_2024_h3.csv")
	if not os.path.exists(crash_path) or not os.path.exists(h3_map_path):
		raise FileNotFoundError("CRASH_2024.csv or CRASH_2024_h3.csv not found")
	crash = pd.read_csv(crash_path)
	# Try to find coordinates
	coords_cols = _detect_lat_lon_columns(crash)
	map_df = pd.read_csv(h3_map_path, usecols=["CRN", "H3_R8"], dtype={"CRN": str, "H3_R8": str})
	# Merge to ensure CRN linkage
	crash["CRN"] = crash["CRN"].astype(str) if "CRN" in crash.columns else crash.index.astype(str)
	crash = crash.merge(map_df, on="CRN", how="left")

	# If H3-level factor, make sure it's present (derive if needed for flags/optionals)
	if from_h3:
		if factor not in h3_df.columns:
			# Try to enrich with optional derivations and flags
			h3_df = add_optional_factors(h3_df)
			h3_df = ensure_flag_factor(h3_df, factor)
		if factor not in h3_df.columns:
			raise ValueError(f"Factor {factor} not found in H3 aggregates even after derivation attempts.")
		crash = crash.merge(h3_df[["H3_R8", factor]], on="H3_R8", how="left", suffixes=("", "_H3"))
		vals = crash[factor]
	else:
		# Crash-level factor
		if factor not in crash.columns:
			raise ValueError(f"Crash-level factor {factor} not found in CRASH_2024.csv")
		# Coerce numeric where possible
		crash[factor] = pd.to_numeric(crash[factor], errors="coerce")
		vals = crash[factor]

	mask = pd.to_numeric(vals, errors="coerce") > float(threshold)
	crash_filt = crash[mask].copy()
	if crash_filt.empty:
		print(f"No crashes where {factor} > {threshold}.")
		return

	# Coordinates: prefer native lat/lon, else fallback to H3 tile centers
	try:
		import h3
	except Exception:
		h3 = None

	if coords_cols:
		lat_col, lon_col = coords_cols
		lat = pd.to_numeric(crash_filt[lat_col], errors="coerce")
		lon = pd.to_numeric(crash_filt[lon_col], errors="coerce")
		coord_mask = (~lat.isna()) & (~lon.isna())
		lat = lat[coord_mask]
		lon = lon[coord_mask]
	elif h3 is not None and "H3_R8" in crash_filt.columns:
		centers = crash_filt["H3_R8"].dropna().astype(str).apply(lambda h: pd.Series(h3.cell_to_latlng(h), index=["lat", "lon"]))
		lat = centers["lat"]
		lon = centers["lon"]
	else:
		raise RuntimeError("No coordinates available to plot crash dots.")

	# Plot
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.scatter(lon, lat, s=float(size), c=color, alpha=0.85, edgecolors="none")
	ax.set_aspect("equal", adjustable="box")
	if minimal:
		ax.axis("off")
	else:
		ax.set_xlabel("Longitude")
		ax.set_ylabel("Latitude")
		title = f"Crashes where {factor} > {threshold}"
		ax.set_title(title)
	# Bounds from points
	try:
		xmin, xmax = float(lon.min()), float(lon.max())
		ymin, ymax = float(lat.min()), float(lat.max())
		ax.set_xlim(xmin, xmax)
		ax.set_ylim(ymin, ymax)
	except Exception:
		pass
	plt.tight_layout()
	def _safe(s: str) -> str:
		return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in s)
	outfile = os.path.join(PLOTS_DIR, f"map_CRASH_DOTS_{_safe(factor)}_gt_{str(threshold).replace('.', 'p')}.png")
	fig.savefig(outfile, dpi=150)
	plt.close(fig)


def plot_h3_clusters(df: pd.DataFrame, label_col: str, minimal: bool = False, title: Optional[str] = None, overlay_centers: bool = True, annotate: bool = False, points_only: bool = False) -> None:
	"""Plot H3 polygons colored by cluster labels (categorical).
	Uses a discrete colormap for clear cluster separation.
	Optionally overlays red dots at spatial centroids of each cluster.
	"""
	ensure_plots_dir()
	try:
		import h3
	except Exception:
		raise RuntimeError("Please install the 'h3' package: pip install h3")

	labels = df[label_col].astype(int)
	unique_labels = sorted(labels.unique().tolist())
	label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
	indices = labels.map(label_to_index).astype(int)

	patches = []
	colors = []
	if not points_only:
		for i in range(len(df)):
			row = df.iloc[i]
			h = row["H3_R8"]
			boundary = h3.cell_to_boundary(h)
			poly = MplPolygon([(lng, lat) for lat, lng in boundary], closed=True)
			patches.append(poly)
			colors.append(int(indices.iloc[i]))

	fig, ax = plt.subplots(figsize=(8, 8))
	if not points_only:
		pc = PatchCollection(patches, cmap="tab20", edgecolor="k", linewidths=0.2)
		pc.set_array(np.array(colors))
		pc.set_clim(vmin=0, vmax=max(colors) if colors else 1)
		ax.add_collection(pc)
	ax.set_aspect("equal", adjustable="box")
	# Overlay cluster spatial centroids
	if overlay_centers:
		# Compute cell centers for each H3, then mean per cluster label
		try:
			import h3
			# Build dataframe with centers and labels
			centers = []
			for i in range(len(df)):
				h = df.iloc[i]["H3_R8"]
				lab = int(df.iloc[i][label_col])
				lat, lng = h3.cell_to_latlng(h)
				centers.append((lab, lat, lng))
			cdf = pd.DataFrame(centers, columns=["label", "lat", "lng"])
			cmeans = cdf.groupby("label").mean(numeric_only=True).reset_index()
			ax.scatter(cmeans["lng"], cmeans["lat"], s=36, c="red", marker="o", edgecolors="k", linewidths=0.7, zorder=3)
			if annotate:
				for _, r in cmeans.iterrows():
					ax.text(float(r["lng"]), float(r["lat"]) , str(int(r["label"])), color="red", fontsize=8, ha="center", va="bottom")
		except Exception:
			pass
	if minimal:
		ax.axis('off')
	else:
		ax.set_xlabel("Longitude")
		ax.set_ylabel("Latitude")
		title = title or (f"KMeans Cluster Centers: {label_col}" if points_only else f"H3 KMeans Clusters: {label_col}")
		ax.set_title(title)
		if not points_only:
			cbar = fig.colorbar(pc, ax=ax)
			cbar.set_ticks(list(range(len(unique_labels))))
			cbar.set_ticklabels([str(l) for l in unique_labels])
	# Auto bounds
	if not points_only and patches:
		all_points = np.vstack([np.array(p.get_xy()) for p in patches])
		ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
		ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
	elif overlay_centers:
		# Bounds from centers if only points
		try:
			xmin, xmax = float(cmeans["lng"].min()), float(cmeans["lng"].max())
			ymin, ymax = float(cmeans["lat"].min()), float(cmeans["lat"].max())
			ax.set_xlim(xmin, xmax)
			ax.set_ylim(ymin, ymax)
		except Exception:
			pass
	plt.tight_layout()
	suffix = "points" if points_only else ""
	out = os.path.join(PLOTS_DIR, f"map_{label_col}{('_' + suffix) if suffix else ''}.png")
	fig.savefig(out, dpi=150)
	plt.close(fig)


def kmeans_cluster_map(h3_df: pd.DataFrame, factors: List[str], k: int = 5, minimal: bool = False, scale: bool = True, annotate: bool = False, include_coords: bool = False, coord_weight: float = 0.3) -> None:
	"""Run KMeans clustering on selected H3 factors and plot cluster map.
	Prints cluster-level means for guidance and saves map to plots/.
	"""
	try:
		from sklearn.cluster import KMeans
		from sklearn.preprocessing import StandardScaler
	except Exception:
		raise RuntimeError("Please install scikit-learn: pip install scikit-learn")

	# Ensure requested factors exist; attempt FLAG derivation if needed
	for f in factors:
		h3_df = ensure_flag_factor(h3_df, f)
	missing = [f for f in factors if f not in h3_df.columns]
	if missing:
		raise ValueError(f"Missing factors for KMeans: {missing}")

	df = h3_df[["H3_R8"] + factors].dropna()
	if df.empty:
		raise ValueError("No rows available for KMeans after dropping NaNs.")
	X = df[factors].astype(float).values
	extra = None
	if include_coords:
		try:
			import h3
			coords = [h3.cell_to_latlng(h) for h in df["H3_R8"]]
			lat = np.array([c[0] for c in coords], dtype=float)
			lng = np.array([c[1] for c in coords], dtype=float)
			extra = np.column_stack([lat, lng])
		except Exception:
			extra = None
	if scale:
		scaler = StandardScaler()
		X = scaler.fit_transform(X)
		if include_coords and extra is not None:
			cscaler = StandardScaler()
			extra_scaled = cscaler.fit_transform(extra) * float(coord_weight)
			X = np.column_stack([X, extra_scaled])
	kmeans = KMeans(n_clusters=int(k), n_init=10, random_state=42)
	labels = kmeans.fit_predict(X)
	df["KMEANS_CLUSTER"] = labels

	# Cluster summaries
	cluster_summary = df.groupby("KMEANS_CLUSTER").mean(numeric_only=True)
	print("KMeans cluster means:")
	print(cluster_summary.to_string())

	# Plot clusters
	label_name = f"KMEANS_{'_'.join(factors)}_K{k}"
	plot_h3_clusters(
		df[["H3_R8", "KMEANS_CLUSTER"]].rename(columns={"KMEANS_CLUSTER": label_name}),
		label_col=label_name,
		minimal=minimal,
		title=f"KMeans (k={k}) on {' + '.join(factors)}",
		overlay_centers=True,
		annotate=annotate,
		points_only=False,
	)


def dbscan_hotspots_map(h3_df: pd.DataFrame, factor: Optional[str] = None, eps_km: float = 5.0, min_samples: int = 10, minimal: bool = False) -> None:
	"""Cluster H3 tiles spatially using DBSCAN and plot cluster centers.
	Optionally filter tiles by a `factor` (e.g., top quantiles of CRASH_RATE).
	eps is in kilometers using haversine distance.
	"""
	try:
		import h3
		from sklearn.cluster import DBSCAN
	except Exception:
		raise RuntimeError("Please install scikit-learn: pip install scikit-learn")

	df = h3_df[["H3_R8"]].copy()
	if factor and (factor in h3_df.columns):
		df[factor] = h3_df[factor].astype(float)
		# Focus on higher values: keep top quartile by default
		try:
			q = df[factor].quantile(0.75)
			df = df[df[factor] >= q]
		except Exception:
			pass
	if df.empty:
		raise ValueError("No tiles available for DBSCAN after filtering.")

	coords = [h3.cell_to_latlng(h) for h in df["H3_R8"]]
	lat = np.array([c[0] for c in coords], dtype=float)
	lng = np.array([c[1] for c in coords], dtype=float)
	# Haversine expects radians; eps in km
	earth_radius_km = 6371.0088
	X = np.column_stack([np.radians(lat), np.radians(lng)])
	db = DBSCAN(eps=float(eps_km) / earth_radius_km, min_samples=int(min_samples), metric="haversine")
	labels = db.fit_predict(X)
	df["DBSCAN_CLUSTER"] = labels
	# Keep only real clusters (label >= 0)
	clusters = df[df["DBSCAN_CLUSTER"] >= 0]
	if clusters.empty:
		print("No spatial clusters found with DBSCAN; try increasing --dbscan-eps-km or lowering --dbscan-min-samples.")
		return
	# Compute cluster centers
	centers = clusters.groupby("DBSCAN_CLUSTER").agg({}).reset_index()
	# manual aggregation to keep clear
	center_rows = []
	for lab in sorted(clusters["DBSCAN_CLUSTER"].unique()):
		mask = clusters["DBSCAN_CLUSTER"] == lab
		latc = float(lat[mask].mean())
		lngc = float(lng[mask].mean())
		center_rows.append({"label": int(lab), "lat": latc, "lng": lngc})
	cdf = pd.DataFrame(center_rows)

	# Plot: base gray tiles plus red center dots
	ensure_plots_dir()
	patches = []
	try:
		for h in h3_df["H3_R8"]:
			boundary = h3.cell_to_boundary(h)
			patches.append(MplPolygon([(lngp, latp) for latp, lngp in boundary], closed=True))
	except Exception:
		pass
	fig, ax = plt.subplots(figsize=(8, 8))
	if patches:
		pc = PatchCollection(patches, facecolor="#f0f0f0", edgecolor="#cccccc", linewidths=0.15)
		ax.add_collection(pc)
	ax.scatter(cdf["lng"], cdf["lat"], s=36, c="red", marker="o", edgecolors="k", linewidths=0.7, zorder=3)
	ax.set_aspect("equal", adjustable="box")
	if minimal:
		ax.axis("off")
	else:
		ax.set_xlabel("Longitude")
		ax.set_ylabel("Latitude")
		ax.set_title(f"DBSCAN Hotspots (eps={eps_km} km, min_samples={min_samples})")
	# Auto bounds
	if patches:
		all_points = np.vstack([np.array(p.get_xy()) for p in patches])
		ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
		ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
	plt.tight_layout()
	out = os.path.join(PLOTS_DIR, f"map_DBSCAN_hotspots.png")
	fig.savefig(out, dpi=150)
	plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
	parser = argparse.ArgumentParser(description="Evaluate road factors contributing to crash rates and produce plots.")
	parser.add_argument("--derive-missing", action="store_true", help="Derive alignment and surface rates from raw files when not present in aggregates.")
	parser.add_argument("--save-csv", action="store_true", help="Save an enriched H3 dataset with derived factors.")
	parser.add_argument("--map-factor", type=str, help="Plot an H3 choropleth map for a given factor column (e.g., ICY_ROAD_RATE).")
	parser.add_argument("--map-quantiles", type=int, default=None, help="Quantile-scale the map colors using the given number of bins (e.g., 10 for deciles).")
	parser.add_argument("--map-threshold", type=float, default=None, help="Only plot H3 cells where the factor exceeds this threshold (e.g., 0.5).")
	parser.add_argument("--map-minimal", action="store_true", help="Render the map without axes or legend for a clean choropleth.")
	parser.add_argument("--map-hotspot", type=str, default=None, help="Plot hotspots where both CRASH_RATE and the given factor exceed their global means (e.g., INTERSECTION_RATE).")
	parser.add_argument("--kmeans-factors", type=str, default=None, help="Comma-separated list of H3 factor columns to cluster (e.g., CRASH_RATE,INTERSECTION_RATE).")
	parser.add_argument("--kmeans-k", type=int, default=5, help="Number of clusters for KMeans.")
	parser.add_argument("--kmeans-minimal", action="store_true", help="Render the KMeans map without axes or legend.")
	parser.add_argument("--kmeans-no-scale", action="store_true", help="Disable feature scaling before KMeans.")
	parser.add_argument("--kmeans-annotate", action="store_true", help="Annotate cluster centers with cluster IDs.")
	parser.add_argument("--kmeans-include-coords", action="store_true", help="Include H3 lat/lng as features to spread clusters spatially.")
	parser.add_argument("--kmeans-coord-weight", type=float, default=0.3, help="Weight applied to scaled coordinates when included.")
	parser.add_argument("--kmeans-points-only", action="store_true", help="Render only KMeans centroid dots (no polygons).")
	parser.add_argument("--dbscan-factor", type=str, default=None, help="Optional factor to filter tiles before DBSCAN (e.g., CRASH_RATE).")
	parser.add_argument("--dbscan-eps-km", type=float, default=5.0, help="DBSCAN neighborhood radius in kilometers.")
	parser.add_argument("--dbscan-min-samples", type=int, default=10, help="Minimum samples per cluster for DBSCAN.")
	parser.add_argument("--dbscan-minimal", action="store_true", help="Render DBSCAN output without axes or legend.")
	# GLM predictions
	parser.add_argument("--predict", action="store_true", help="Fit GLM with offset and write per-H3 predictions.")
	parser.add_argument("--predict-factors", type=str, default=None, help="Comma-separated factor columns to include in prediction model.")
	parser.add_argument("--predict-output", type=str, default=os.path.join(AGG_DIR, "h3_predictions.csv"), help="Output CSV for predictions.")
	parser.add_argument("--evaluate", action="store_true", help="Compute evaluation metrics (MAE, RMSE, Poisson deviance) on a holdout.")
	parser.add_argument("--eval-factors", type=str, default=None, help="Comma-separated factor columns for evaluation model.")
	parser.add_argument("--eval-test-frac", type=float, default=0.2, help="Fraction of tiles for test split.")
	parser.add_argument("--eval-no-spatial", action="store_true", help="Disable spatial block splitting even if available.")
	# Crash dots by threshold
	parser.add_argument("--crash-dots-factor", type=str, default=None, help="Factor to filter crashes by (H3-level or crash-level).")
	parser.add_argument("--crash-dots-threshold", type=float, default=None, help="Threshold value; crashes above this are plotted as dots.")
	parser.add_argument("--crash-dots-from-crash", action="store_true", help="Interpret factor as a crash-level column in CRASH_2024.csv; otherwise use H3-level factor.")
	parser.add_argument("--crash-dots-color", type=str, default="red", help="Dot color (e.g., red, black, #RRGGBB).")
	parser.add_argument("--crash-dots-size", type=float, default=6.0, help="Dot size for plotted crashes.")
	parser.add_argument("--crash-dots-minimal", action="store_true", help="Render crash-dots map without axes or legend.")
	parser.add_argument("--flag-dark-analysis", action="store_true", help="Analyze predictiveness of FLAGS_2024.ILLUMINATION_DARK for crashes at H3 level.")
	parser.add_argument("--age-injury", action="store_true", help="Analyze impact of driver age buckets on INJURY_COUNT using crash-level regression.")
	parser.add_argument("--age-summary", action="store_true", help="Summarize which age groups have the most crashes and highest injury rates.")
	args = parser.parse_args(argv)

	ensure_plots_dir()
	h3 = load_h3_aggregates()
	label_map = load_label_maps()

	# Optionally derive alignment/surface
	if args.derive_missing:
		h3 = add_optional_factors(h3)

	# Collect factors including optional derived ones
	factors = ROAD_FACTOR_COLUMNS.copy()
	for opt_col in [
		"RDWY_ALIGNMENT_CURVE_RATE",
		"RDWY_SURF_ASPHALT_RATE",
		"RDWY_SURF_CONCRETE_RATE",
		"RDWY_SURF_GRAVEL_RATE",
		"RDWY_SURF_OTHER_RATE",
		"ILLUM_DAYLIGHT_RATE",
		"ILLUM_DARK_LIGHTED_RATE",
		"ILLUM_DARK_UNLIGHTED_RATE",
		"ILLUM_OTHER_RATE",
	]:
		if opt_col in h3.columns:
			factors.append(opt_col)

	# Plot relationships
	plot_factor_relationships(h3, factors, label_map)

	# GLM with offset
	fit_glm_and_plot(h3, factors)

	# Optionally save enriched dataset
	if args.save_csv:
		out_csv = os.path.join(AGG_DIR, "h3_modeling_dataset_enriched.csv")
		h3.to_csv(out_csv, index=False)
		print(f"Saved enriched dataset to {out_csv}")

	# Optional choropleth map
	if args.map_factor:
		factor = args.map_factor
		# Try to derive FLAG_* factors on demand
		h3 = ensure_flag_factor(h3, factor)
		if factor not in h3.columns:
			print(f"Factor {factor} not found in dataset.")
		else:
			try:
				df_map = h3[["H3_R8", factor]].dropna()
				if args.map_threshold is not None:
					try:
						thr = float(args.map_threshold)
						df_map = df_map[df_map[factor] > thr]
						if df_map.empty:
							print(f"No H3 cells where {factor} > {thr}.")
						else:
							plot_h3_choropleth(df_map, factor, quantiles=args.map_quantiles, minimal=args.map_minimal)
					except Exception as e:
						print(f"Invalid threshold or plotting error: {e}")
				else:
							plot_h3_choropleth(df_map, factor, quantiles=args.map_quantiles, minimal=args.map_minimal)
			except Exception as e:
				print(f"Failed to plot H3 choropleth for {factor}: {e}")

	# Hotspot map where CRASH_RATE and factor exceed global means
	if args.map_hotspot:
		factor = args.map_hotspot
		# Try to derive FLAG_* factors on demand
		h3 = ensure_flag_factor(h3, factor)
		if factor not in h3.columns:
			print(f"Factor {factor} not found in dataset.")
		else:
			try:
				df_map = h3[["H3_R8", "CRASH_RATE", factor]].dropna()
				cr_mean = float(df_map["CRASH_RATE"].mean())
				fx_mean = float(df_map[factor].mean())
				hot = df_map[(df_map["CRASH_RATE"] > cr_mean) & (df_map[factor] > fx_mean)]
				if hot.empty:
					print(f"No hotspots found where CRASH_RATE and {factor} exceed their means.")
				else:
					# Color by crash rate quantiles for contrast
					plot_h3_choropleth(hot[["H3_R8", "CRASH_RATE"]].rename(columns={"CRASH_RATE": f"HOT_{factor}"}), f"HOT_{factor}", quantiles=10, minimal=args.map_minimal)
			except Exception as e:
				print(f"Failed to plot hotspot map for {factor}: {e}")

	# Flag analysis: ILLUMINATION_DARK
	if args.flag_dark_analysis:
		try:
			analyze_flag_illumination_dark()
		except Exception as e:
			print(f"Failed ILLUMINATION_DARK flag analysis: {e}")

	# Crash-level analysis of age vs injury count
	if args.age_injury:
		try:
			analyze_age_vs_injury()
		except Exception as e:
			print(f"Failed age vs injury analysis: {e}")

	# Crash-level age group summary (counts and injury rates)
	if args.age_summary:
		try:
			age_group_summary()
		except Exception as e:
			print(f"Failed age group summary: {e}")

	# KMeans clustering on selected factors
	if args.kmeans_factors:
		try:
			factors = [f.strip() for f in args.kmeans_factors.split(",") if f.strip()]
			if not factors:
				raise ValueError("No valid factors provided for KMeans.")
			kmeans_cluster_map(h3, factors, k=args.kmeans_k, minimal=args.kmeans_minimal, scale=(not args.kmeans_no_scale), annotate=args.kmeans_annotate, include_coords=args.kmeans_include_coords, coord_weight=args.kmeans_coord_weight)
			# If points-only requested, also emit a centers-only map
			if args.kmeans_points_only:
				try:
					label_name = f"KMEANS_{'_'.join(factors)}_K{args.kmeans_k}"
					# Recompute to access centers quickly via existing function
					# Use minimal duplication: call kmeans_cluster_map with points_only=True
					kmeans_cluster_map(h3, factors, k=args.kmeans_k, minimal=True, scale=(not args.kmeans_no_scale), annotate=args.kmeans_annotate, include_coords=args.kmeans_include_coords, coord_weight=args.kmeans_coord_weight)
				except Exception as e:
					print(f"Failed KMeans points-only render: {e}")
		except Exception as e:
			print(f"Failed KMeans clustering: {e}")

	# DBSCAN clustering (spatial hotspots)
	if args.dbscan_factor or (args.dbscan_factor is None):
		try:
			dbscan_hotspots_map(h3, factor=args.dbscan_factor, eps_km=args.dbscan_eps_km, min_samples=args.dbscan_min_samples, minimal=args.dbscan_minimal)
		except Exception as e:
			print(f"Failed DBSCAN hotspots: {e}")
	# GLM predictions
	if args.predict:
		try:
			if args.predict_factors:
				pred_factors = [f.strip() for f in args.predict_factors.split(",") if f.strip()]
			else:
				# Default to core road factors present
				pred_factors = [c for c in ROAD_FACTOR_COLUMNS if c in h3.columns]
			# Include derived optional factors if present
			for opt in [
				"RDWY_ALIGNMENT_CURVE_RATE",
				"RDWY_SURF_ASPHALT_RATE",
				"RDWY_SURF_CONCRETE_RATE",
				"RDWY_SURF_GRAVEL_RATE",
				"RDWY_SURF_OTHER_RATE",
				"ILLUM_DAYLIGHT_RATE",
				"ILLUM_DARK_LIGHTED_RATE",
				"ILLUM_DARK_UNLIGHTED_RATE",
				"ILLUM_OTHER_RATE",
				"FLAG_ILLUMINATION_DARK_RATE",
			]:
				if opt in h3.columns:
					pred_factors.append(opt)
			pred_df = predict_glm_counts(h3, pred_factors)
			pred_df.to_csv(args.predict_output, index=False)
			print(f"Saved GLM predictions to {args.predict_output}")
		except Exception as e:
			print(f"Failed GLM predictions: {e}")

	# GLM evaluation
	if args.evaluate:
		try:
			if args.eval_factors:
				eval_factors = [f.strip() for f in args.eval_factors.split(",") if f.strip()]
			else:
				eval_factors = [c for c in ROAD_FACTOR_COLUMNS if c in h3.columns]
			for opt in [
				"RDWY_ALIGNMENT_CURVE_RATE",
				"RDWY_SURF_ASPHALT_RATE",
				"RDWY_SURF_CONCRETE_RATE",
				"RDWY_SURF_GRAVEL_RATE",
				"RDWY_SURF_OTHER_RATE",
				"ILLUM_DAYLIGHT_RATE",
				"ILLUM_DARK_LIGHTED_RATE",
				"ILLUM_DARK_UNLIGHTED_RATE",
				"ILLUM_OTHER_RATE",
				"FLAG_ILLUMINATION_DARK_RATE",
			]:
				if opt in h3.columns:
					eval_factors.append(opt)
			metrics = evaluate_glm(h3, eval_factors, test_frac=float(args.eval_test_frac), use_spatial_blocks=(not args.eval_no_spatial))
			print("Evaluation metrics:")
			for k, v in metrics.items():
				print(f" - {k}: {v:.4f}")
		except Exception as e:
			print(f"Failed GLM evaluation: {e}")

	# Crash dots filtered by threshold
	if args.crash_dots_factor and (args.crash_dots_threshold is not None):
		try:
			plot_crash_dots_threshold(
				h3_df=h3,
				factor=args.crash_dots_factor,
				threshold=float(args.crash_dots_threshold),
				from_h3=(not args.crash_dots_from_crash),
				color=args.crash_dots_color,
				size=float(args.crash_dots_size),
				minimal=args.crash_dots_minimal,
			)
		except Exception as e:
			print(f"Failed crash-dots map: {e}")



def analyze_age_vs_injury() -> None:
	"""Crash-level regression: INJURY_COUNT ~ driver age bucket counts.
	Generates a coefficients bar plot (IRR if Poisson, else exp(beta)) saved under plots/age_injury_effects.png.
	"""
	crash_path = os.path.join(DATA_DIR, "CRASH_2024.csv")
	if not os.path.exists(crash_path):
		raise FileNotFoundError("CRASH_2024.csv not found")
	cols = [
		"INJURY_COUNT",
		"DRIVER_COUNT_16YR",
		"DRIVER_COUNT_17YR",
		"DRIVER_COUNT_18YR",
		"DRIVER_COUNT_19YR",
		"DRIVER_COUNT_20YR",
		"DRIVER_COUNT_50_64YR",
		"DRIVER_COUNT_65_74YR",
		"DRIVER_COUNT_75PLUS",
	]
	df = pd.read_csv(crash_path, usecols=cols)
	# Clean numeric
	for c in cols:
		df[c] = pd.to_numeric(df[c], errors="coerce")
	df = df.dropna()
	# Remove rows with all zero drivers to avoid degenerate fits
	age_cols = cols[1:]
	if (df[age_cols].sum(axis=1) == 0).any():
		df = df[(df[age_cols].sum(axis=1) > 0)]
	# Design matrix: use raw counts and include total drivers as control
	X = df[age_cols].copy()
	total_drivers = X.sum(axis=1)
	X["TOTAL_DRIVERS"] = total_drivers
	y = df["INJURY_COUNT"].astype(float)
	# Fit GLM Poisson if available
	if sm is None:
		raise RuntimeError("statsmodels is required for regression analysis")
	X = sm.add_constant(X)
	# Use Negative Binomial for stability
	model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
	res = model.fit()
	params = res.params.drop("const", errors="ignore")
	conf = res.conf_int().loc[params.index]
	irr = np.exp(params)
	irr_lo = np.exp(conf[0])
	irr_hi = np.exp(conf[1])
	df_plot = pd.DataFrame({
		"age_bucket": params.index,
		"IRR": irr,
		"IRR_lo": irr_lo,
		"IRR_hi": irr_hi,
	}).sort_values("IRR", ascending=False)
	# Remove TOTAL_DRIVERS from the IRR plot to focus on age buckets
	df_plot = df_plot[df_plot["age_bucket"] != "TOTAL_DRIVERS"]
	# Guard against extreme values for visualization
	df_plot["IRR"] = df_plot["IRR"].clip(upper=10)
	df_plot["IRR_lo"] = df_plot["IRR_lo"].clip(lower=0, upper=10)
	df_plot["IRR_hi"] = df_plot["IRR_hi"].clip(upper=10)
	# Plot
	ensure_plots_dir()
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.barh(df_plot["age_bucket"], df_plot["IRR"], xerr=[df_plot["IRR"] - df_plot["IRR_lo"], df_plot["IRR_hi"] - df_plot["IRR"]], color="#9467bd", alpha=0.85)
	ax.invert_yaxis()
	ax.set_xlabel("Incidence Rate Ratio (IRR) on INJURY_COUNT")
	ax.set_title("Effect of Driver Age Buckets on Injury Count (Crash-level)")
	# Ensure reasonable x-limits
	if not df_plot.empty:
		xmax = float(df_plot["IRR_hi"].max())
		ax.set_xlim(left=0, right=max(1.0, xmax))
	plt.tight_layout()
	out = os.path.join(PLOTS_DIR, "age_injury_effects.png")
	fig.savefig(out, dpi=150)
	plt.close(fig)
	# Print summary
	print("Age vs Injury IRR summary:")
	for _, row in df_plot.iterrows():
		print(f" - {row['age_bucket']}: IRR={row['IRR']:.3f} (CI {row['IRR_lo']:.3f}–{row['IRR_hi']:.3f})")


def age_group_summary() -> None:
	"""Summarize which age groups have the most crashes and highest injury rates.
	We treat driver age bucket columns as counts per crash and aggregate across crashes.
	- Crash involvement per age group: sum of driver counts
	- Injury involvement per age group: sum of driver counts for crashes with INJURY_COUNT > 0
	- Injury rate per age group: injury_involvement / crash_involvement
	Generates two bar charts under plots/age_group_crash_counts.png and plots/age_group_injury_rates.png
	"""
	crash_path = os.path.join(DATA_DIR, "CRASH_2024.csv")
	if not os.path.exists(crash_path):
		raise FileNotFoundError("CRASH_2024.csv not found")
	age_cols = [
		"DRIVER_COUNT_16YR",
		"DRIVER_COUNT_17YR",
		"DRIVER_COUNT_18YR",
		"DRIVER_COUNT_19YR",
		"DRIVER_COUNT_20YR",
		"DRIVER_COUNT_50_64YR",
		"DRIVER_COUNT_65_74YR",
		"DRIVER_COUNT_75PLUS",
	]
	cols = ["INJURY_COUNT"] + age_cols
	df = pd.read_csv(crash_path, usecols=cols)
	# Coerce numeric
	for c in cols:
		df[c] = pd.to_numeric(df[c], errors="coerce")
	df = df.fillna(0)
	# Injury indicator
	df["HAS_INJURY"] = (df["INJURY_COUNT"] > 0).astype(int)
	# Aggregate crash involvement (sum of driver counts per age bucket)
	crash_involvement = df[age_cols].sum(axis=0)
	# Aggregate injury involvement: sum age counts for injured crashes
	injury_involvement = df.loc[df["HAS_INJURY"] == 1, age_cols].sum(axis=0)
	# Injury rate per age bucket
	injury_rate = (injury_involvement / crash_involvement.replace(0, np.nan)).fillna(0)
	# Prepare plotting
	ensure_plots_dir()
	# Bar chart: crash involvement counts
	fig1, ax1 = plt.subplots(figsize=(9, 5))
	ax1.barh(list(crash_involvement.index.astype(str)), list(crash_involvement.astype(float).values), color="#1f77b4")
	ax1.invert_yaxis()
	ax1.set_xlabel("Crash Involvement (sum of driver counts)")
	ax1.set_title("Driver Age Groups: Crash Involvement")
	plt.tight_layout()
	fig1.savefig(os.path.join(PLOTS_DIR, "age_group_crash_counts.png"), dpi=150)
	plt.close(fig1)
	# Bar chart: injury rates
	fig2, ax2 = plt.subplots(figsize=(9, 5))
	ax2.barh(list(injury_rate.index.astype(str)), list(injury_rate.astype(float).values), color="#d62728")
	ax2.invert_yaxis()
	ax2.set_xlabel("Injury Rate (INJURY_COUNT > 0)")
	ax2.set_title("Driver Age Groups: Injury Rates")
	ax2.set_xlim(left=0, right=max(0.01, float(injury_rate.max()) * 1.1))
	plt.tight_layout()
	fig2.savefig(os.path.join(PLOTS_DIR, "age_group_injury_rates.png"), dpi=150)
	plt.close(fig2)
	# Print top groups
	print("Top age groups by crash involvement:")
	print(crash_involvement.sort_values(ascending=False).to_string())
	print("\nTop age groups by injury rate:")
	print(injury_rate.sort_values(ascending=False).to_string())


if __name__ == "__main__":
	main()

