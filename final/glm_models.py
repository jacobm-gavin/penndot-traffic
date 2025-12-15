import os
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception:
    sm = None

from paths import AGG_DIR, PLOTS_DIR, ensure_plots_dir


def fit_glm_and_plot(h3_df: pd.DataFrame, factors: List[str]) -> None:
    """Fit Poisson/NB GLM with offset and plot IRR bar chart."""
    ensure_plots_dir()
    cols = [c for c in factors if c in h3_df.columns]
    X = h3_df[cols].copy()
    y = h3_df["CRASH_COUNT"].astype(float)
    offset = h3_df["LOG_AADT"].astype(float)
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
    df_plot = pd.DataFrame({"factor": params.index, "IRR": irr, "IRR_lo": irr_lo, "IRR_hi": irr_hi}).sort_values("IRR", ascending=False)
    import matplotlib.pyplot as plt

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
    """Fit a Poisson GLM with offset(LOG_AADT) and return predictions."""
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
    eta = res.predict(X_const, offset=offset, linear=True)
    mu = np.exp(eta)
    rate = np.exp(eta - offset)
    pred_df = pd.DataFrame({"H3_R8": h3_df.loc[idx, "H3_R8"].values, "PRED_CRASH_COUNT": mu, "PRED_CRASH_RATE": rate})
    return pred_df


def evaluate_glm(h3_df: pd.DataFrame, factors: List[str], test_frac: float = 0.2, use_spatial_blocks: bool = True) -> Dict[str, float]:
    """Evaluate GLM with optional spatial block split."""
    if sm is None:
        raise RuntimeError("statsmodels not available; cannot evaluate GLM.")
    cols = [c for c in factors if c in h3_df.columns]
    if not cols:
        raise ValueError("No valid factor columns found for evaluation.")
    df = h3_df[["H3_R8", "CRASH_COUNT", "LOG_AADT"] + cols].dropna()
    if df.empty:
        raise ValueError("No rows available for evaluation after dropping NaNs.")
    train_idx = None
    test_idx = None
    blocks_path = os.path.join(AGG_DIR, "spatial_blocks.csv")
    if use_spatial_blocks and os.path.exists(blocks_path):
        try:
            blocks = pd.read_csv(blocks_path)
            if {"H3_R8", "block_id"}.issubset(set(blocks.columns)):
                dfb = df.merge(blocks[["H3_R8", "block_id"]], on="H3_R8", how="left")
                unique_blocks = dfb["block_id"].dropna().unique()
                if len(unique_blocks) > 0:
                    np.random.seed(42)
                    test_blocks = set(np.random.choice(unique_blocks, size=max(1, int(len(unique_blocks) * test_frac)), replace=False))
                    train_idx = dfb.index[~dfb["block_id"].isin(test_blocks)]
                    test_idx = dfb.index[dfb["block_id"].isin(test_blocks)]
        except Exception:
            train_idx = None
            test_idx = None
    if train_idx is None or test_idx is None or len(test_idx) == 0:
        np.random.seed(42)
        perm = np.random.permutation(len(df))
        n_test = max(1, int(len(df) * test_frac))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
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
    X_test = sm.add_constant(test[cols])
    off_test = test["LOG_AADT"].astype(float)
    eta_test = res.predict(X_test, offset=off_test, linear=True)
    mu_test = np.exp(eta_test)
    y_true = np.asarray(test["CRASH_COUNT"].astype(float).values)
    mae = float(np.mean(np.abs(y_true - mu_test)))
    rmse = float(np.sqrt(np.mean((y_true - mu_test) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y_true > 0.0, y_true * np.log(y_true / np.asarray(mu_test)), 0.0)
        dev = 2.0 * np.sum(term - (y_true - np.asarray(mu_test)))
    return {"MAE": mae, "RMSE": rmse, "PoissonDeviance": float(dev)}
