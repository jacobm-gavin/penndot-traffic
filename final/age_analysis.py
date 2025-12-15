import os

import numpy as np
import pandas as pd

import statsmodels.api as sm

from paths import DATA_DIR, PLOTS_DIR, ensure_plots_dir


def analyze_age_vs_injury() -> None:
    """Crash-level regression: INJURY_COUNT ~ driver age bucket counts."""
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
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    age_cols = cols[1:]
    if (df[age_cols].sum(axis=1) == 0).any():
        df = df[(df[age_cols].sum(axis=1) > 0)]
    X = df[age_cols].copy()
    total_drivers = X.sum(axis=1)
    X["TOTAL_DRIVERS"] = total_drivers
    y = df["INJURY_COUNT"].astype(float)
    X = sm.add_constant(X)
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    res = model.fit()
    params = res.params.drop("const", errors="ignore")
    conf = res.conf_int().loc[params.index]
    irr = np.exp(params)
    irr_lo = np.exp(conf[0])
    irr_hi = np.exp(conf[1])
    df_plot = pd.DataFrame({"age_bucket": params.index, "IRR": irr, "IRR_lo": irr_lo, "IRR_hi": irr_hi}).sort_values("IRR", ascending=False)
    df_plot = df_plot[df_plot["age_bucket"] != "TOTAL_DRIVERS"]
    df_plot["IRR"] = df_plot["IRR"].clip(upper=10)
    df_plot["IRR_lo"] = df_plot["IRR_lo"].clip(lower=0, upper=10)
    df_plot["IRR_hi"] = df_plot["IRR_hi"].clip(upper=10)
    import matplotlib.pyplot as plt

    ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_plot["age_bucket"], df_plot["IRR"], xerr=[df_plot["IRR"] - df_plot["IRR_lo"], df_plot["IRR_hi"] - df_plot["IRR"]], color="#9467bd", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Incidence Rate Ratio (IRR) on INJURY_COUNT")
    ax.set_title("Effect of Driver Age Buckets on Injury Count (Crash-level)")
    if not df_plot.empty:
        xmax = float(df_plot["IRR_hi"].max())
        ax.set_xlim(left=0, right=max(1.0, xmax))
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "age_injury_effects.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Age vs Injury IRR summary:")
    for _, row in df_plot.iterrows():
        print(f" - {row['age_bucket']}: IRR={row['IRR']:.3f} (CI {row['IRR_lo']:.3f}–{row['IRR_hi']:.3f})")


def age_group_summary() -> None:
    """Summarize which age groups have the most crashes and highest injury rates."""
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
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0)
    df["HAS_INJURY"] = (df["INJURY_COUNT"] > 0).astype(int)
    crash_involvement = df[age_cols].sum(axis=0)
    injury_involvement = df.loc[df["HAS_INJURY"] == 1, age_cols].sum(axis=0)
    injury_rate = (injury_involvement / crash_involvement.replace(0, np.nan)).fillna(0)
    ensure_plots_dir()
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.barh(list(crash_involvement.index.astype(str)), list(crash_involvement.astype(float).values), color="#1f77b4")
    ax1.invert_yaxis()
    ax1.set_xlabel("Crash Involvement (sum of driver counts)")
    ax1.set_title("Driver Age Groups: Crash Involvement")
    plt.tight_layout()
    fig1.savefig(os.path.join(PLOTS_DIR, "age_group_crash_counts.png"), dpi=150)
    plt.close(fig1)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.barh(list(injury_rate.index.astype(str)), list(injury_rate.astype(float).values), color="#d62728")
    ax2.invert_yaxis()
    ax2.set_xlabel("Injury Rate (INJURY_COUNT > 0)")
    ax2.set_title("Driver Age Groups: Injury Rates")
    ax2.set_xlim(left=0, right=max(0.01, float(injury_rate.max()) * 1.1))
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOTS_DIR, "age_group_injury_rates.png"), dpi=150)
    plt.close(fig2)
    print("Top age groups by crash involvement:")
    print(crash_involvement.sort_values(ascending=False).to_string())
    print("\nTop age groups by injury rate:")
    print(injury_rate.sort_values(ascending=False).to_string())
