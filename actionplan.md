# Action Plan: Crash Contributing Factors in H3 Grid (2024 PennDOT)

## Objectives
- Identify which roadway, environmental, and contextual factors are most associated with higher crash frequency across grid cells.
- Provide interpretable insights (effect sizes, SHAP) and actionable rankings of higher-risk grid areas.

## Data Sources & Keys
- Crash-level: `data/CRASH_2024.csv` (location, time, severity, counts, environment).
- Flags: `data/FLAGS_2024.csv` (engineered binary indicators aligned to CRN).
- Roadway: `data/ROADWAY_2024.csv` (segment context; keys include `ROUTE`, `SEGMENT`, `SPEED_LIMIT`, `LANE_COUNT`, `ACCESS_CTRL`, `RAMP`, `RDWY_ORIENT`).
- Units/Vehicles: `data/VEHICLE_2024.csv` (unit-level; aggregate to crash-level then to grid).
- Persons: `data/PERSON_2024.csv` (person-level; use for EDA, avoid outcome leakage for severity labels).
- Commercial/Motorcycle/Trailer: `COMMVEH_2024.csv`, `CYCLE_2024.csv`, `TRAILVEH_2024.csv` (use for flags at crash-level).
- AADT spatial data (CSV-like): road segments with attributes:
  - Keys/fields: `RMSTRAFFIC_LRS_KEY` (PK), `ST_RT_NO` (state route), `CURR_AADT` (int total volume), `TRAFF_PATT_GRP` (pattern group: interstate, rural, etc.).
  - Geometry: prefer a WKT `geometry` column (LineString/MultiLineString). If absent, a companion centerline geometry keyed by `RMSTRAFFIC_LRS_KEY` or `ST_RT_NO` is required to compute segment length per H3 cell.
- Primary key: `CRN`. Joins for enrichment at crash-level (then spatially aggregate to grid).

Note: Per dictionary, `ROUTE` = route number (state roads), `SEGMENT` = segment number, `ACCESS_CTRL` = access control code, etc. (Use `src/dictionary.txt` via grep to map code values when needed.)

## Problem Framing
- Unit of analysis: H3 hexagonal grid at resolution 8 (~0.74 km² per cell).
- Target (primary): Crash count per grid cell in 2024 (optionally severity-weighted EPDO).
- Secondary target (optional): Serious vs other, derived from `MAX_SEVERITY_LEVEL` at crash level, then aggregated for EDA or separate modeling.
- Normalization/exposure: Offset = log(exposure), where grid exposure = Σ over AADT segments intersecting the cell of (`CURR_AADT` × segment_length_within_cell × 365). No lane-weighting (lane count not available). Include `TRAFF_PATT_GRP` as a categorical feature to capture facility type.

## Data Preparation
1) Ingest
- Read all CSVs with explicit dtypes; trim header whitespace; standardize column names to uppercase.
- Verify one-to-one CRN cardinality for `CRASH_2024` and `FLAGS_2024`.

2) Grid design and spatialization
- Use H3 resolution 8. For each crash in `CRASH_2024`, compute `H3_8` from `DEC_LATITUDE/DEC_LONGITUDE`.
- Keep crash-level attributes needed for later aggregation (time, environment, flags via join on `CRN`).

3) AADT exposure aggregation to H3
- Parse AADT CSV: if WKT geometry exists, load with GeoPandas (`from_wkt`), set CRS, project to a suitable planar CRS (e.g., EPSG:26918). If no geometry column, join to a centerline geometry source via `RMSTRAFFIC_LRS_KEY` or `ST_RT_NO`.
- Build H3 res 8 cell polygons. Intersect AADT polylines with H3 cells; compute `length_within_cell` (meters).
- Compute exposure per cell: `exposure_cell = Σ (CURR_AADT × length_within_cell_m × 365)`; keep units consistent across all cells.
- Retain `TRAFF_PATT_GRP` distribution per cell (exposure-weighted shares or majority) as predictors.

4) Grid-level feature engineering
- Environment/context: aggregate shares by `WEATHER1/2`, `ILLUMINATION`, `ROAD_CONDITION`, `RDWY_SURF_TYPE_CD`.
- Intersections/controls: shares by `INTERSECTION_RELATED`, `INTERSECT_TYPE`, `TCD_TYPE`, `TCD_FUNC_CD`.
- Temporal mix: crash shares by `CRASH_MONTH`, `DAY_OF_WEEK`, `HOUR_OF_DAY`, `TIME_OF_DAY`.
- Work zones: include `WORK_ZONE_IND`, `WORK_ZONE_TYPE`, `WORK_ZONE_LOC`, `WZ_*` as predictors; exclude `WZ_WORKERS_INJ_KILLED`.
- Vehicle mix: from `VEHICLE_2024`, aggregate per grid cell (heavy trucks, motorcycles, buses, pedestrians/bicycles involvement).
- Spatial indicators: county/municipality prevalence from `CRASH_2024`, `URBAN_RURAL` mix.
- Road class/exposure context: add `TRAFF_PATT_GRP` features per cell (exposure-weighted shares; optionally collapse rare groups).
- Encode categoricals (one-hot or target-agnostic impact coding with CV); cap rare levels.

5) Severity derivation (for optional severity modeling/EDA)
- From dictionary (MAX_SEVERITY_LEVEL): 0=PDO, 1=Fatal, 2=Suspected Serious, 3=Suspected Minor, 4=Possible, 8=Injury-Unknown, 9=Unknown.
- Define `SERIOUS = 1 if MAX_SEVERITY_LEVEL in {1,2} else 0`. Aggregate cell-level serious share/counts for analysis (not used as predictors in severity model).

6) Leakage control
- Do not use outcome-derived fields as predictors: `MAX_SEVERITY_LEVEL`, `FATAL_COUNT`, `INJURY_COUNT`, `TOT_INJ_COUNT`, `UNB_DEATH_COUNT`, `WZ_WORKERS_INJ_KILLED`.
- Exclude response-time fields (`ARRIVAL_TM`, `DISPATCH_TM`) and any post-crash variables.

## Modeling
A) Grid crash frequency (primary)
- Baseline: Negative Binomial GLM with log link.
  - Formula (example): `crash_count_cell ~ urban_rural_mix + intersection_share + work_zone_share + weather_mix + illumination_mix + vehicle_mix + county/muni indicators + s(speed_limit_mix)`.
  - Offset: `log(exposure_cell)` from AADT × length × 365 (lane-weighted optional).
- Alternatives: Zero-inflated NB (if many zero-crash cells), Poisson-Gamma hierarchical (Empirical Bayes), or gradient boosting (Poisson/Tweedie) for ranking and SHAP interpretation.
- Interpretation: Coefficients as rate ratios; partial dependence for continuous effects; SHAP for boosted models.

B) Crash severity (optional)
- Label: `SERIOUS = 1{MAX_SEVERITY_LEVEL ∈ {1,2}}` at crash-level; aggregate features to cell-level for modeling or analyze at crash-level with spatial CV.
- Models: Logistic regression with regularization, GAMs; Gradient boosting with probability calibration.
- Use for factor interpretation and to compare with frequency model drivers.

## Validation & Evaluation
- Spatial CV: hold out by county or by route-blocks to reduce spatial leakage.
- If more years become available: temporal split (train prev., test 2024).
- Diagnostics: check overdispersion (justify NB vs Poisson), consider zero-inflation (Vuong), residual spatial autocorrelation (Moran’s I).
- Metrics: deviance/AIC for GLMs; PR@K for hotspot ranking; calibration curves for severity.
- Stability checks: rank correlation of feature importances and hotspot lists across folds.

## Implementation Plan (Folders/Scripts)
- `01_ingest.py`: Read CSVs, schema enforcement, basic QA.
- `02_assign_h3.py`: Compute H3 res 8 index (`H3_8`) for each crash; export crash+H3 data.
- `03_overlay_aadt_h3.py`: Load AADT CSV, parse WKT geometry (or join to centerlines), intersect with H3; compute within-cell length and `exposure_cell` using `CURR_AADT`; aggregate `TRAFF_PATT_GRP` shares per cell.
- `05_features_grid.py`: Aggregate crash/context/vehicle/work zone features to grid level.
- `06_model_nb_grid.py`: Fit NB GLM; export coefficients, effect sizes, diagnostics.
- `07_model_gbm_grid.py`: Train Poisson/Tweedie GBM; SHAP summaries; partial dependence.
- `08_eval_grid.py`: Spatial CV (county/cell blocks), hotspot PR@K, calibration; export reports.

## Deliverables
- Clean grid-level dataset with dictionary of engineered features and exposure per cell.
- Model artifacts: GLM results, SHAP plots, partial dependence.
- Ranked H3 cells and (optionally) maps of predicted risk.
- Model card documenting assumptions, validation, and limitations.

## Open Items
1) AADT data path/format (e.g., Shapefile/GeoJSON/Geopackage) and CRS; confirm if lane count and segment length are present or must be computed.
2) Additional years: if available, extend to a panel for temporal validation.
3) Tooling preferences: any preferred modeling stack (statsmodels/GLM, XGBoost/LightGBM)?

## Next Steps
- Confirm grid type/size and AADT dataset location.
- Implement `01–05` scripts to build grid dataset with exposure and features.
- Fit baseline NB model with AADT exposure; check overdispersion/zero-inflation; iterate features.
- Add boosted model for SHAP-based factor ranking and compare insights.
