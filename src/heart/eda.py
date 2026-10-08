"""Exploratory data analysis - writes figures to reports/figures/eda.

Run:  python -m heart.eda
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart import config  # noqa: E402
from heart.data import load_clean  # noqa: E402

log = logging.getLogger(__name__)
OUT = config.FIGURES_DIR / "eda"
PALETTE = {0: "#2471a3", 1: "#c0392b"}
LABELS = {0: "No disease", 1: "Disease"}

DESCRIPTIONS = {
    "age": "Age (years)", "trestbps": "Resting BP (mm Hg)", "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate", "oldpeak": "ST depression", "ca": "Major vessels (0-3)",
    "sex": "Sex (1 = male)", "fbs": "Fasting sugar > 120", "exang": "Exercise angina",
    "cp": "Chest pain type", "restecg": "Resting ECG", "slope": "ST slope", "thal": "Thalassemia",
}


def _save(fig, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return str(path)


def class_balance(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    counts = df[config.TARGET].map(LABELS).value_counts()
    axes[0].bar(counts.index, counts.values, color=[PALETTE[1], PALETTE[0]]
                if counts.index[0] == "Disease" else [PALETTE[0], PALETTE[1]])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 8, f"{v}  ({v / len(df):.0%})", ha="center")
    axes[0].set_title("Overall class balance")
    axes[0].set_ylabel("patients")
    site = pd.crosstab(df["site"], df[config.TARGET].map(LABELS))
    site.plot.barh(stacked=True, ax=axes[1], color=[PALETTE[1], PALETTE[0]])
    axes[1].set_title("Class balance by collection site")
    axes[1].set_xlabel("patients")
    axes[1].legend(title=None)
    fig.tight_layout()
    return _save(fig, "01_class_balance.png")


def missingness(df: pd.DataFrame) -> str:
    miss = df[config.FEATURES].isna().groupby(df["site"]).mean().T * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(miss, annot=True, fmt=".0f", cmap="Oranges", cbar_kws={"label": "% missing"}, ax=ax)
    ax.set_title("Missing values (%) by feature and site")
    ax.set_xlabel("")
    return _save(fig, "02_missingness_by_site.png")


def numeric_histograms(df: pd.DataFrame) -> str:
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(cols), figsize=(20, 3.8))
    for ax, col in zip(axes, cols, strict=True):
        sns.histplot(data=df, x=col, hue=config.TARGET, palette=PALETTE, kde=True,
                     element="step", stat="density", common_norm=False, ax=ax, legend=False)
        ax.set_title(DESCRIPTIONS[col])
        ax.set_xlabel("")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[k]) for k in (0, 1)]
    fig.legend(handles, LABELS.values(), loc="upper right", ncol=2)
    fig.suptitle("Numeric feature distributions by diagnosis", y=1.04, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_numeric_histograms.png")


def categorical_rates(df: pd.DataFrame) -> str:
    cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, col in zip(axes.ravel(), cols, strict=True):
        s = df[col].fillna(-1).astype(int).astype(str).replace("-1", "missing")
        rate = df.groupby(s)[config.TARGET].agg(["mean", "size"])
        ax.bar(rate.index, rate["mean"] * 100, color="#7d3c98")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"], strict=True)):
            ax.text(i, m * 100 + 2, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_title(DESCRIPTIONS[col])
        ax.set_ylabel("% with disease")
    fig.suptitle("Disease rate per category level", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_categorical_disease_rate.png")


def correlation(df: pd.DataFrame) -> str:
    corr = df[config.FEATURES + [config.TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman correlation (pairwise complete)")
    return _save(fig, "05_correlation_heatmap.png")


def boxplots(df: pd.DataFrame) -> str:
    cols = ["age", "thalach", "oldpeak", "chol"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    d = df.assign(diagnosis=df[config.TARGET].map(LABELS))
    for ax, col in zip(axes, cols, strict=True):
        sns.boxplot(data=d, x="diagnosis", y=col, hue="diagnosis",
                    palette={LABELS[k]: v for k, v in PALETTE.items()}, ax=ax, legend=False)
        ax.set_title(DESCRIPTIONS[col])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    return _save(fig, "06_boxplots_outliers.png")


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    desc = df[config.FEATURES].describe().T
    desc["missing"] = df[config.FEATURES].isna().sum()
    desc["missing_%"] = (desc["missing"] / len(df) * 100).round(1)
    out = config.ROOT / "reports" / "eda_summary.csv"
    desc.round(2).to_csv(out)
    return desc


def run_all() -> list[str]:
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_clean()
    summary_table(df)
    return [f(df) for f in (class_balance, missingness, numeric_histograms,
                            categorical_rates, correlation, boxplots)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_all()
