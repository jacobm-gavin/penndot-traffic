import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import yaml

from utils.modeling import FeatureConfig, build_design_matrix


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_glm(df: pd.DataFrame, cfg_dict: Dict[str, Any], outdir: str) -> None:
    cfg = FeatureConfig(
        response=cfg_dict["response"],
        offset=cfg_dict.get("offset"),
        include=cfg_dict.get("include", []),
        exclude=cfg_dict.get("exclude", []),
        categorical=cfg_dict.get("categorical", []),
        min_cat_freq=float(cfg_dict.get("min_cat_freq", 0.01)),
    )

    X, y, offset = build_design_matrix(df, cfg)

    # Drop rows with missing response (and offset if provided); X missing already filled in build step
    mask_valid = y.notna()
    if offset is not None:
        mask_valid &= offset.notna()
    X = X.loc[mask_valid]
    y = y.loc[mask_valid]
    off_arr = None

    model_type = cfg_dict.get("model", {}).get("type", "glm_nb")

    try:
        import statsmodels.api as sm
        from statsmodels.genmod.families import Poisson, NegativeBinomial
    except Exception as e:
        print("statsmodels not available; cannot train GLM.", file=sys.stderr)
        raise

    fam = NegativeBinomial() if model_type == "glm_nb" else Poisson()
    if offset is not None:
        # statsmodels expects offset on linear predictor scale (log exposure)
        import numpy as np
        off_arr = np.log(offset.loc[mask_valid].clip(lower=1e-6).astype(float))

    if X.shape[1] == 0:
        raise ValueError("No predictor columns selected. Check config include/exclude patterns or fallback selection.")
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y.astype(float), X_const.astype(float), family=fam, offset=off_arr)
    results = model.fit()

    os.makedirs(outdir, exist_ok=True)
    # Save summary and coefficients
    with open(os.path.join(outdir, "glm_summary.txt"), "w") as f:
        f.write(results.summary().as_text())
    coef_df = pd.DataFrame({"feature": ["const"] + X.columns.tolist(), "coef": results.params.values})
    coef_df.to_csv(os.path.join(outdir, "glm_coefficients.csv"), index=False)

    # Save metadata
    meta = {
        "model_type": model_type,
        "response": cfg.response,
        "offset": cfg.offset,
        "n_obs": int(results.nobs),
        "aic": float(results.aic),
        "bic": float(getattr(results, "bic", float("nan"))),
        "converged": bool(results.mle_retvals.get("converged", True)) if hasattr(results, "mle_retvals") else True,
        "features": X.columns.tolist(),
    }
    with open(os.path.join(outdir, "glm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Trained {model_type} GLM on {int(results.nobs)} rows; artifacts in {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Train GLM model for crash frequency with offset")
    parser.add_argument("--features", default="data/aggregates/h3_features.csv", help="Input features CSV")
    parser.add_argument("--config", default="config/model_config.yaml", help="Model config YAML")
    parser.add_argument("--outdir", default="model_artifacts/glm", help="Output directory for artifacts")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    cfg = load_config(args.config)
    train_glm(df, cfg, args.outdir)


if __name__ == "__main__":
    main()
