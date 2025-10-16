import importlib.util
from pathlib import Path
import pandas as pd


def test_build_features_handles_empty_exposure(tmp_path: Path):
    # Minimal crash base with required columns
    crash = pd.DataFrame({
        'CRN': [1, 2, 3],
        'DEC_LATITUDE': [40.0, 40.1, 40.2],
        'DEC_LONGITUDE': [-75.0, -75.1, -75.2],
        'MAX_SEVERITY_LEVEL': [0, 1, 3],
        'URBAN_RURAL': [1, 1, 0],
        'INTERSECTION_RELATED': [0, 1, 0],
        'WORK_ZONE_IND': [0, 0, 1],
        'ILLUMINATION': [1, 2, 1],
        'ROAD_CONDITION': [1, 1, 2],
        'RDWY_SURF_TYPE_CD': [1, 1, 1],
        'WEATHER1': [1, 1, 2],
        'WEATHER2': [0, 0, 0],
        'VEHICLE_COUNT': [2, 1, 3],
        'HEAVY_TRUCK_COUNT': [0, 1, 0],
    })
    crash_h3 = pd.DataFrame({
        'CRN': [1, 2, 3],
        'H3_R8': ['hexA', 'hexA', 'hexB']
    })
    # Empty exposure file
    exposure_path = tmp_path / 'exposure.csv'
    exposure_path.write_text("")

    crash_path = tmp_path / 'crash.csv'
    crash_h3_path = tmp_path / 'crash_h3.csv'
    out_path = tmp_path / 'features.csv'
    crash.to_csv(crash_path, index=False)
    crash_h3.to_csv(crash_h3_path, index=False)

    # Dynamically load build_features from script with numeric prefix
    mod_path = Path(__file__).resolve().parents[1] / '05_build_h3_features.py'
    spec = importlib.util.spec_from_file_location('build_h3_features', str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    # Should not raise even with empty exposure
    mod.build_features(str(crash_path), str(crash_h3_path), str(exposure_path), str(out_path))
    df = pd.read_csv(out_path)
    assert 'EXPOSURE_AADT_MXM_365' in df.columns
    assert (df['EXPOSURE_AADT_MXM_365'] >= 1.0).all()
