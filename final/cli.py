import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# Allow importing sibling modules when run as a script
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
	sys.path.insert(0, str(THIS_DIR))

from age_analysis import age_group_summary, analyze_age_vs_injury
from clustering import dbscan_hotspots_map, kmeans_cluster_map
from constants import ROAD_FACTOR_COLUMNS
from data_sources import load_h3_aggregates, load_label_maps
from derivations import add_optional_factors
from flags import analyze_flag_illumination_dark, ensure_flag_factor
from glm_models import evaluate_glm, fit_glm_and_plot, predict_glm_counts
from paths import AGG_DIR, ensure_plots_dir
from plotting import plot_crash_dots_threshold, plot_factor_relationships, plot_h3_choropleth


def _collect_optional_factors(h3) -> List[str]:
	opt_cols = [
		"RDWY_ALIGNMENT_CURVE_RATE",
		"RDWY_SURF_ASPHALT_RATE",
		"RDWY_SURF_CONCRETE_RATE",
		"RDWY_SURF_GRAVEL_RATE",
		"RDWY_SURF_OTHER_RATE",
		"ILLUM_DAYLIGHT_RATE",
		"ILLUM_DARK_LIGHTED_RATE",
		"ILLUM_DARK_UNLIGHTED_RATE",
		"ILLUM_OTHER_RATE",
	]
	return [c for c in opt_cols if c in h3.columns]


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
	parser.add_argument("--predict", action="store_true", help="Fit GLM with offset and write per-H3 predictions.")
	parser.add_argument("--predict-factors", type=str, default=None, help="Comma-separated factor columns to include in prediction model.")
	parser.add_argument("--predict-output", type=str, default=os.path.join(AGG_DIR, "h3_predictions.csv"), help="Output CSV for predictions.")
	parser.add_argument("--evaluate", action="store_true", help="Compute evaluation metrics (MAE, RMSE, Poisson deviance) on a holdout.")
	parser.add_argument("--eval-factors", type=str, default=None, help="Comma-separated factor columns for evaluation model.")
	parser.add_argument("--eval-test-frac", type=float, default=0.2, help="Fraction of tiles for test split.")
	parser.add_argument("--eval-no-spatial", action="store_true", help="Disable spatial block splitting even if available.")
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
	if args.derive_missing:
		h3 = add_optional_factors(h3)
	factors = ROAD_FACTOR_COLUMNS.copy()
	for opt_col in _collect_optional_factors(h3):
		factors.append(opt_col)
	plot_factor_relationships(h3, factors, label_map)
	fit_glm_and_plot(h3, factors)
	if args.save_csv:
		out_csv = os.path.join(AGG_DIR, "h3_modeling_dataset_enriched.csv")
		h3.to_csv(out_csv, index=False)
		print(f"Saved enriched dataset to {out_csv}")
	if args.map_factor:
		factor = args.map_factor
		h3 = ensure_flag_factor(h3, factor)
		if factor not in h3.columns:
			print(f"Factor {factor} not found in dataset.")
		else:
			try:
				df_map = h3[["H3_R8", factor]].dropna()
				if args.map_threshold is not None:
					thr = float(args.map_threshold)
					df_map = df_map[df_map[factor] > thr]
					if df_map.empty:
						print(f"No H3 cells where {factor} > {thr}.")
					else:
						plot_h3_choropleth(df_map, factor, quantiles=args.map_quantiles, minimal=args.map_minimal)
				else:
					plot_h3_choropleth(df_map, factor, quantiles=args.map_quantiles, minimal=args.map_minimal)
			except Exception as e:
				print(f"Failed to plot H3 choropleth for {factor}: {e}")
	if args.map_hotspot:
		factor = args.map_hotspot
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
					plot_h3_choropleth(hot[["H3_R8", "CRASH_RATE"]].rename(columns={"CRASH_RATE": f"HOT_{factor}"}), f"HOT_{factor}", quantiles=10, minimal=args.map_minimal)
			except Exception as e:
				print(f"Failed to plot hotspot map for {factor}: {e}")
	if args.flag_dark_analysis:
		try:
			analyze_flag_illumination_dark()
		except Exception as e:
			print(f"Failed ILLUMINATION_DARK flag analysis: {e}")
	if args.age_injury:
		try:
			analyze_age_vs_injury()
		except Exception as e:
			print(f"Failed age vs injury analysis: {e}")
	if args.age_summary:
		try:
			age_group_summary()
		except Exception as e:
			print(f"Failed age group summary: {e}")
	if args.kmeans_factors:
		try:
			factors_km = [f.strip() for f in args.kmeans_factors.split(",") if f.strip()]
			if not factors_km:
				raise ValueError("No valid factors provided for KMeans.")
			kmeans_cluster_map(
				h3,
				factors_km,
				k=args.kmeans_k,
				minimal=args.kmeans_minimal,
				scale=(not args.kmeans_no_scale),
				annotate=args.kmeans_annotate,
				include_coords=args.kmeans_include_coords,
				coord_weight=args.kmeans_coord_weight,
				points_only=False,
			)
			if args.kmeans_points_only:
				try:
					kmeans_cluster_map(
						h3,
						factors_km,
						k=args.kmeans_k,
						minimal=True,
						scale=(not args.kmeans_no_scale),
						annotate=args.kmeans_annotate,
						include_coords=args.kmeans_include_coords,
						coord_weight=args.kmeans_coord_weight,
						points_only=True,
					)
				except Exception as e:
					print(f"Failed KMeans points-only render: {e}")
		except Exception as e:
			print(f"Failed KMeans clustering: {e}")
	if args.dbscan_factor or (args.dbscan_factor is None):
		try:
			dbscan_hotspots_map(h3, factor=args.dbscan_factor, eps_km=args.dbscan_eps_km, min_samples=args.dbscan_min_samples, minimal=args.dbscan_minimal)
		except Exception as e:
			print(f"Failed DBSCAN hotspots: {e}")
	if args.predict:
		try:
			if args.predict_factors:
				pred_factors = [f.strip() for f in args.predict_factors.split(",") if f.strip()]
			else:
				pred_factors = [c for c in ROAD_FACTOR_COLUMNS if c in h3.columns]
			for opt in _collect_optional_factors(h3) + ["FLAG_ILLUMINATION_DARK_RATE"]:
				if opt in h3.columns:
					pred_factors.append(opt)
			pred_df = predict_glm_counts(h3, pred_factors)
			pred_df.to_csv(args.predict_output, index=False)
			print(f"Saved GLM predictions to {args.predict_output}")
		except Exception as e:
			print(f"Failed GLM predictions: {e}")
	if args.evaluate:
		try:
			if args.eval_factors:
				eval_factors = [f.strip() for f in args.eval_factors.split(",") if f.strip()]
			else:
				eval_factors = [c for c in ROAD_FACTOR_COLUMNS if c in h3.columns]
			for opt in _collect_optional_factors(h3) + ["FLAG_ILLUMINATION_DARK_RATE"]:
				if opt in h3.columns:
					eval_factors.append(opt)
			metrics = evaluate_glm(h3, eval_factors, test_frac=float(args.eval_test_frac), use_spatial_blocks=(not args.eval_no_spatial))
			print("Evaluation metrics:")
			for k, v in metrics.items():
				print(f" - {k}: {v:.4f}")
		except Exception as e:
			print(f"Failed GLM evaluation: {e}")
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


if __name__ == "__main__":
	main()
