import argparse
import math
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Polygon, Point
from shapely.ops import transform as shp_transform

try:
    import h3
except Exception as e:
    print("Missing dependency: h3. Please install via `pip install h3`.", file=sys.stderr)
    raise

try:
    from pyproj import Transformer
except Exception as e:
    print("Missing dependency: pyproj. Please install via `pip install pyproj`.", file=sys.stderr)
    raise


WGS84 = "EPSG:4326"
PA_UTM = "EPSG:26918"  # NAD83 / UTM zone 18N (covers most of PA)


def get_transformers():
    to_proj = Transformer.from_crs(WGS84, PA_UTM, always_xy=True)
    to_geo = Transformer.from_crs(PA_UTM, WGS84, always_xy=True)

    def proj_fn(x, y, z=None):
        x2, y2 = to_proj.transform(x, y)
        return (x2, y2) if z is None else (x2, y2, z)

    def geo_fn(x, y, z=None):
        x2, y2 = to_geo.transform(x, y)
        return (x2, y2) if z is None else (x2, y2, z)

    return proj_fn, geo_fn


def _cell_from_latlon(lat: float, lon: float, res: int) -> Optional[str]:
    try:
        return h3.geo_to_h3(lat, lon, res)
    except AttributeError:
        try:
            return h3.latlng_to_cell(lat, lon, res)
        except Exception:
            return None


def h3_hex_polygon(cell: str) -> Polygon:
    try:
        boundary = h3.h3_to_geo_boundary(cell)  # v3
    except AttributeError:
        boundary = h3.cell_to_boundary(cell)  # v4
    # boundary is sequence of (lat, lon); shapely expects (lon, lat)
    coords = [(lon, lat) for (lat, lon) in boundary]
    return Polygon(coords)


def densify_candidates(line_proj: LineString, step_m: float = 50.0) -> List[str]:
    # Sample along projected line every step_m, map to H3 in WGS84, deduplicate
    if line_proj.length == 0:
        return []
    proj_fn, geo_fn = get_transformers()
    num = max(1, int(math.ceil(line_proj.length / step_m)))
    seen: set = set()
    cells: List[str] = []
    for i in range(num + 1):
        d = min(line_proj.length, i * step_m)
        pt = line_proj.interpolate(d)
        x, y = pt.x, pt.y
        lon, lat = geo_fn(x, y)
        cell = _cell_from_latlon(lat, lon, 8)
        if cell is None:
            continue
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)
    return cells


def line_length_in_cell_m(line_wgs: LineString, cell: str) -> float:
    proj_fn, geo_fn = get_transformers()
    # Project line and cell polygon to planar CRS
    line_proj = shp_transform(proj_fn, line_wgs)
    hex_poly_wgs = h3_hex_polygon(cell)
    hex_poly_proj = shp_transform(proj_fn, hex_poly_wgs)
    inter = line_proj.intersection(hex_poly_proj)
    return float(inter.length) if not inter.is_empty else 0.0


def _detect_aadt_col(columns_upper: List[str]) -> Optional[str]:
    candidates = [
        "CURR_AADT",
        "CUR_AADT",
        "AADT",
        "CURR_AADT_TOTAL",
        "BASE_ADT",
    ]
    for c in candidates:
        if c in columns_upper:
            return c
    return None


def _detect_patt_col(columns_upper: List[str]) -> Optional[str]:
    candidates = [
        "TRAFF_PATT_GRP",
        "TRAFF_PATT_GROUP",
        "TRAFFIC_PATTERN",
        "TRAFF_PATT",
    ]
    for c in candidates:
        if c in columns_upper:
            return c
    return None


def _sanitize_name(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "UNKNOWN"


def compute_exposure(aadt_csv: str, out_csv: str, geometry_col: str = "GEOMETRY", chunksize: Optional[int] = 50_000) -> None:
    # We'll detect AADT and pattern columns dynamically; require geometry column to be present
    first = True
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Aggregate exposure and pattern counts per H3 cell
    exposure: Dict[str, float] = defaultdict(float)
    patt_exp: Dict[str, Counter] = defaultdict(Counter)

    for chunk in pd.read_csv(aadt_csv, chunksize=chunksize):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        if geometry_col.upper() not in chunk.columns:
            raise ValueError(f"Missing geometry column '{geometry_col}' in {aadt_csv}")
        aadt_col = _detect_aadt_col(list(chunk.columns))
        if not aadt_col:
            raise ValueError(
                f"Missing AADT column in {aadt_csv}. Looked for one of: CURR_AADT, CUR_AADT, AADT, CURR_AADT_TOTAL, BASE_ADT. Found: {list(chunk.columns)}"
            )
        patt_col = _detect_patt_col(list(chunk.columns))

        # Coerce types
        chunk[aadt_col] = pd.to_numeric(chunk[aadt_col], errors="coerce")
        chunk = chunk[~chunk[geometry_col.upper()].isna()]
        # Skip rows without geometry or AADT
        chunk = chunk[(chunk[geometry_col.upper()].astype(str).str.len() > 0) & (chunk[aadt_col].notna())]

        for _, row in chunk.iterrows():
            try:
                geom = wkt.loads(row[geometry_col.upper()])
            except Exception:
                continue
            # Split MultiLineString into parts
            parts: List[LineString]
            if isinstance(geom, LineString):
                parts = [geom]
            elif isinstance(geom, MultiLineString):
                parts = [g for g in geom.geoms if isinstance(g, LineString)]
            else:
                continue

            aadt = float(row[aadt_col]) if pd.notna(row[aadt_col]) else 0.0
            patt_val = row[patt_col] if patt_col and (patt_col in row) else "UNKNOWN"
            patt = _sanitize_name(patt_val)
            if aadt <= 0:
                continue

            for part in parts:
                # Compute candidate cells by densifying in projected space
                proj_fn, geo_fn = get_transformers()
                line_proj = shp_transform(proj_fn, part)
                candidates = densify_candidates(line_proj, step_m=50.0)
                if not candidates:
                    continue
                # Precise length allocation by intersection per candidate cell
                for cell in candidates:
                    length_m = line_length_in_cell_m(part, cell)
                    if length_m <= 0:
                        continue
                    exp = aadt * length_m * 365.0
                    exposure[cell] += exp
                    patt_exp[cell][patt] += exp

    # Build output dataframe
    rows = []
    for cell, exp in exposure.items():
        row = {"H3_R8": cell, "EXPOSURE_AADT_MXM_365": exp}
        # Top patterns and shares
        total = sum(patt_exp[cell].values())
        # Include top-5 groups as columns
        for patt, val in patt_exp[cell].most_common(5):
            safe = _sanitize_name(patt)
            row[f"PATT_{safe}_EXPOSURE"] = val
            row[f"PATT_{safe}_SHARE"] = (val / total) if total > 0 else 0.0
        rows.append(row)

    # Ensure header exists even if no rows
    if rows:
        out_df = pd.DataFrame(rows)
    else:
        out_df = pd.DataFrame(columns=["H3_R8", "EXPOSURE_AADT_MXM_365"])  # minimal header
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote H3 exposure to {out_csv} with {len(out_df)} cells")


def main():
    parser = argparse.ArgumentParser(description="Overlay AADT (CSV with WKT geometry) onto H3 res 8 to compute exposure per cell")
    parser.add_argument("--input", default="data/RMSTRAFFIC.csv", help="Path to AADT CSV")
    parser.add_argument("--output", default="data/H3_res8_exposure.csv", help="Path to output exposure CSV")
    parser.add_argument("--geometry-col", default="GEOMETRY", help="Name of WKT geometry column (LineString)")
    args = parser.parse_args()

    compute_exposure(args.input, args.output, geometry_col=args.geometry_col)


if __name__ == "__main__":
    main()
