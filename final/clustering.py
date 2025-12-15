import os

import numpy as np
import pandas as pd

from flags import ensure_flag_factor
from plotting import plot_h3_clusters


def kmeans_cluster_map(
	h3_df: pd.DataFrame,
	factors: list,
	k: int = 5,
	minimal: bool = False,
	scale: bool = True,
	annotate: bool = False,
	include_coords: bool = False,
	coord_weight: float = 0.3,
	points_only: bool = False,
) -> None:
	"""Run KMeans clustering on selected H3 factors and plot cluster map."""
	try:
		from sklearn.cluster import KMeans
		from sklearn.preprocessing import StandardScaler
	except Exception:
		raise RuntimeError("Please install scikit-learn: pip install scikit-learn")
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
	cluster_summary = df.groupby("KMEANS_CLUSTER").mean(numeric_only=True)
	print("KMeans cluster means:")
	print(cluster_summary.to_string())
	label_name = f"KMEANS_{'_'.join(factors)}_K{k}"
	plot_h3_clusters(
		df[["H3_R8", "KMEANS_CLUSTER"]].rename(columns={"KMEANS_CLUSTER": label_name}),
		label_col=label_name,
		minimal=minimal,
		title=f"KMeans (k={k}) on {' + '.join(factors)}",
		overlay_centers=True,
		annotate=annotate,
		points_only=points_only,
	)


def dbscan_hotspots_map(h3_df: pd.DataFrame, factor: str | None = None, eps_km: float = 5.0, min_samples: int = 10, minimal: bool = False) -> None:
	"""Cluster H3 tiles spatially using DBSCAN and plot cluster centers."""
	try:
		import h3
		from sklearn.cluster import DBSCAN
	except Exception:
		raise RuntimeError("Please install scikit-learn: pip install scikit-learn")
	df = h3_df[["H3_R8"]].copy()
	if factor and (factor in h3_df.columns):
		df[factor] = h3_df[factor].astype(float)
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
	earth_radius_km = 6371.0088
	X = np.column_stack([np.radians(lat), np.radians(lng)])
	db = DBSCAN(eps=float(eps_km) / earth_radius_km, min_samples=int(min_samples), metric="haversine")
	labels = db.fit_predict(X)
	df["DBSCAN_CLUSTER"] = labels
	clusters = df[df["DBSCAN_CLUSTER"] >= 0]
	if clusters.empty:
		print("No spatial clusters found with DBSCAN; try increasing --dbscan-eps-km or lowering --dbscan-min-samples.")
		return
	centers = []
	for lab in sorted(clusters["DBSCAN_CLUSTER"].unique()):
		mask = clusters["DBSCAN_CLUSTER"] == lab
		latc = float(lat[mask].mean())
		lngc = float(lng[mask].mean())
		centers.append({"label": int(lab), "lat": latc, "lng": lngc})
	cdf = pd.DataFrame(centers)
	from matplotlib.patches import Polygon as MplPolygon
	from matplotlib.collections import PatchCollection
	import matplotlib.pyplot as plt
	from paths import PLOTS_DIR, ensure_plots_dir

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
	if patches:
		all_points = np.vstack([np.array(p.get_xy()) for p in patches])
		ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
		ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
	plt.tight_layout()
	out = os.path.join(PLOTS_DIR, f"map_DBSCAN_hotspots.png")
	fig.savefig(out, dpi=150)
	plt.close(fig)
