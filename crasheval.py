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
	valid = (~X.isna().any(axis=1)) & (~y.isna()) & (~offset.isna())
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


def main(argv: Optional[List[str]] = None) -> None:
	parser = argparse.ArgumentParser(description="Evaluate road factors contributing to crash rates and produce plots.")
	parser.add_argument("--derive-missing", action="store_true", help="Derive alignment and surface rates from raw files when not present in aggregates.")
	parser.add_argument("--save-csv", action="store_true", help="Save an enriched H3 dataset with derived factors.")
	parser.add_argument("--map-factor", type=str, help="Plot an H3 choropleth map for a given factor column (e.g., ICY_ROAD_RATE).")
	parser.add_argument("--map-quantiles", type=int, default=None, help="Quantile-scale the map colors using the given number of bins (e.g., 10 for deciles).")
	parser.add_argument("--map-threshold", type=float, default=None, help="Only plot H3 cells where the factor exceeds this threshold (e.g., 0.5).")
	parser.add_argument("--map-minimal", action="store_true", help="Render the map without axes or legend for a clean choropleth.")
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

