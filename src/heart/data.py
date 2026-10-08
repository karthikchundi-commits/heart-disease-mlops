"""Data acquisition and cleaning for the UCI Heart Disease dataset."""
from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from heart import config

log = logging.getLogger(__name__)


def download(dest: Path = config.DATA_RAW, force: bool = False) -> list[Path]:
    """Download the four processed site files from the UCI repository."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname in config.SITES.values():
        target = dest / fname
        if target.exists() and not force:
            log.info("cached  %s", target.name)
        else:
            url = config.UCI_BASE + fname
            log.info("fetch   %s", url)
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
                target.write_bytes(resp.read())
        paths.append(target)
    return paths


def load_raw(raw_dir: Path = config.DATA_RAW) -> pd.DataFrame:
    """Read the raw site files into one frame; '?' marks a missing value."""
    frames = []
    for site, fname in config.SITES.items():
        df = pd.read_csv(raw_dir / fname, header=None, names=config.RAW_COLUMNS,
                         na_values="?", skipinitialspace=True)
        df["site"] = site
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the combined raw data.

    * binarise the 0-4 diagnosis into presence (1) / absence (0) of disease
    * physiologically impossible zeros (chol = 0, trestbps = 0) -> missing
    * out-of-dictionary category codes -> missing
    * drop exact duplicate rows
    Remaining missing values are imputed inside the model pipeline so the
    same logic is applied at training and serving time.
    """
    df = df.copy()
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    for col in ("chol", "trestbps"):
        df.loc[df[col] == 0, col] = np.nan

    for col, levels in config.CATEGORY_LEVELS.items():
        df.loc[~df[col].isin(levels), col] = np.nan

    for col in config.BINARY_FEATURES:
        df.loc[~df[col].isin([0, 1]), col] = np.nan

    df = df.drop_duplicates().reset_index(drop=True)
    return df[config.FEATURES + ["site", config.TARGET]]


def load_clean(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or config.CLEAN_CSV)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download and clean the UCI heart disease data")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    download(force=args.force)
    raw = load_raw()
    df = clean(raw)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_CSV, index=False)
    log.info("raw rows=%d  clean rows=%d  positives=%.1f%%  -> %s",
             len(raw), len(df), 100 * df[config.TARGET].mean(), config.CLEAN_CSV)
    log.info("missing values per column:\n%s", df.isna().sum()[df.isna().sum() > 0].to_string())


if __name__ == "__main__":
    main()
