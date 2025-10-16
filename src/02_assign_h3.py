import argparse
import os
import sys
from typing import Optional

import pandas as pd

try:
    import h3
except Exception as e:
    print("Missing dependency: h3. Please install via `pip install h3`.", file=sys.stderr)
    raise


def _cell_from_latlon(lat: float, lon: float, res: int) -> Optional[str]:
    """Compatibility wrapper for h3 v3 (geo_to_h3) and v4 (latlng_to_cell)."""
    try:
        return h3.geo_to_h3(lat, lon, res)  # h3-py v3
    except AttributeError:
        try:
            return h3.latlng_to_cell(lat, lon, res)  # h3-py v4
        except Exception:
            return None


def assign_h3(crash_csv: str, out_csv: str, res: int = 8, chunksize: Optional[int] = 100_000) -> None:
    cols_needed = ["CRN", "DEC_LATITUDE", "DEC_LONGITUDE"]
    first = True
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    for chunk in pd.read_csv(crash_csv, chunksize=chunksize):
        # Normalize column names
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        missing = [c for c in cols_needed if c not in chunk.columns]
        if missing:
            raise ValueError(f"Missing columns in {crash_csv}: {missing}")

        df = chunk[cols_needed].copy()
        # Ensure numeric
        df["DEC_LATITUDE"] = pd.to_numeric(df["DEC_LATITUDE"], errors="coerce")
        df["DEC_LONGITUDE"] = pd.to_numeric(df["DEC_LONGITUDE"], errors="coerce")
        df = df.dropna(subset=["DEC_LATITUDE", "DEC_LONGITUDE"])  # drop records without coordinates

        df["H3_R8"] = [_cell_from_latlon(lat, lon, res) for lat, lon in zip(df["DEC_LATITUDE"].values, df["DEC_LONGITUDE"].values)]
        df = df.dropna(subset=["H3_R8"])

        df.to_csv(out_csv, index=False, mode="w" if first else "a", header=first)
        first = False


def main():
    parser = argparse.ArgumentParser(description="Assign H3 (res 8) to crashes by DEC_LATITUDE/DEC_LONGITUDE")
    parser.add_argument("--input", default="data/CRASH_2024.csv", help="Path to crash CSV")
    parser.add_argument("--output", default="data/CRASH_2024_h3.csv", help="Path to output CSV with H3 index")
    parser.add_argument("--res", type=int, default=8, help="H3 resolution (default 8)")
    args = parser.parse_args()

    assign_h3(args.input, args.output, args.res)
    print(f"Wrote H3-assigned crashes to {args.output}")


if __name__ == "__main__":
    main()
