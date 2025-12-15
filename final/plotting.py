import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

from constants import readable_label
from derivations import add_optional_factors
from flags import ensure_flag_factor
from paths import PLOTS_DIR, DATA_DIR, ensure_plots_dir


def plot_factor_relationships(h3_df: pd.DataFrame, factors: List[str], label_map: dict) -> None:
    """Scatter crash rate vs each factor with optional binned mean overlay."""
    ensure_plots_dir()
    sns.set(style="whitegrid")
    for col in factors:
        if col not in h3_df.columns:
            continue
        x = h3_df[col]
        y = h3_df["CRASH_RATE"]
        valid = (~x.isna()) & (~y.isna())
        x = x[valid]
        y = y[valid]
        if len(x) < 10:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, s=12, alpha=0.35, color="#1f77b4")
        try:
            q = pd.qcut(x, q=10, duplicates="drop")
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


def plot_h3_choropleth(df: pd.DataFrame, factor: str, quantiles: Optional[int] = None, minimal: bool = False) -> None:
    """Plot H3 polygons colored by factor value (optionally quantile-scaled)."""
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
            q = pd.qcut(vals, q=quantiles, labels=False, duplicates="drop")
            vals_scaled = q.astype(float)
            vmin, vmax = 0.0, float(vals_scaled.max())
            color_values = vals_scaled
            colorbar_label = f"{factor} (quantile rank)"
        except Exception:
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
        boundary = h3.cell_to_boundary(h)
        poly = MplPolygon([(lng, lat) for lat, lng in boundary], closed=True)
        patches.append(poly)
        if quantiles and quantiles > 1 and "vals_scaled" in locals():
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
        ax.axis("off")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"H3 Choropleth: {factor}")
        fig.colorbar(pc, ax=ax, label=colorbar_label)
    all_points = np.vstack([np.array(p.get_xy()) for p in patches])
    ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"map_{factor}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _detect_lat_lon_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Heuristically detect latitude/longitude column names in crash-level data."""
    cols = {c.lower(): c for c in df.columns}
    lat_candidates = ["latitude", "lat", "dec_lat", "y", "gps_lat", "crash_latitude"]
    lon_candidates = ["longitude", "lon", "lng", "dec_long", "x", "gps_long", "crash_longitude"]
    lat_col = next((cols[c] for c in lat_candidates if c in cols), None)
    lon_col = next((cols[c] for c in lon_candidates if c in cols), None)
    if lat_col and lon_col:
        return lat_col, lon_col
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
    """Plot crash-level dots for crashes where a factor exceeds a threshold."""
    ensure_plots_dir()
    crash_path = os.path.join(DATA_DIR, "CRASH_2024.csv")
    h3_map_path = os.path.join(DATA_DIR, "CRASH_2024_h3.csv")
    if not os.path.exists(crash_path) or not os.path.exists(h3_map_path):
        raise FileNotFoundError("CRASH_2024.csv or CRASH_2024_h3.csv not found")
    crash = pd.read_csv(crash_path)
    coords_cols = _detect_lat_lon_columns(crash)
    map_df = pd.read_csv(h3_map_path, usecols=["CRN", "H3_R8"], dtype={"CRN": str, "H3_R8": str})
    crash["CRN"] = crash["CRN"].astype(str) if "CRN" in crash.columns else crash.index.astype(str)
    crash = crash.merge(map_df, on="CRN", how="left")
    if from_h3:
        if factor not in h3_df.columns:
            h3_df = add_optional_factors(h3_df)
            h3_df = ensure_flag_factor(h3_df, factor)
        if factor not in h3_df.columns:
            raise ValueError(f"Factor {factor} not found in H3 aggregates even after derivation attempts.")
        crash = crash.merge(h3_df[["H3_R8", factor]], on="H3_R8", how="left", suffixes=("", "_H3"))
        vals = crash[factor]
    else:
        if factor not in crash.columns:
            raise ValueError(f"Crash-level factor {factor} not found in CRASH_2024.csv")
        crash[factor] = pd.to_numeric(crash[factor], errors="coerce")
        vals = crash[factor]
    mask = pd.to_numeric(vals, errors="coerce") > float(threshold)
    crash_filt = crash[mask].copy()
    if crash_filt.empty:
        print(f"No crashes where {factor} > {threshold}.")
        return
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
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(lon, lat, s=float(size), c=color, alpha=0.85, edgecolors="none")
    ax.set_aspect("equal", adjustable="box")
    if minimal:
        ax.axis("off")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Crashes where {factor} > {threshold}")
    try:
        xmin, xmax = float(lon.min()), float(lon.max())
        ymin, ymax = float(lat.min()), float(lat.max())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    except Exception:
        pass
    plt.tight_layout()

    def _safe(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)

    outfile = os.path.join(PLOTS_DIR, f"map_CRASH_DOTS_{_safe(factor)}_gt_{str(threshold).replace('.', 'p')}.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_h3_clusters(df: pd.DataFrame, label_col: str, minimal: bool = False, title: Optional[str] = None, overlay_centers: bool = True, annotate: bool = False, points_only: bool = False) -> None:
    """Plot H3 polygons colored by cluster labels (categorical)."""
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
    if overlay_centers:
        try:
            import h3
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
                    ax.text(float(r["lng"]), float(r["lat"]), str(int(r["label"])), color="red", fontsize=8, ha="center", va="bottom")
        except Exception:
            pass
    if minimal:
        ax.axis("off")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        title = title or (f"KMeans Cluster Centers: {label_col}" if points_only else f"H3 KMeans Clusters: {label_col}")
        ax.set_title(title)
        if not points_only:
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_ticks(list(range(len(unique_labels))))
            cbar.set_ticklabels([str(l) for l in unique_labels])
    if not points_only and patches:
        all_points = np.vstack([np.array(p.get_xy()) for p in patches])
        ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
        ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    elif overlay_centers:
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
