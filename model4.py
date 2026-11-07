"""One-command Ridge48 model: python model4.py."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json

import pandas as pd
import matplotlib
from utils import features as ridge
from utils.experiment import run_experiment

ROOT = Path(__file__).resolve().parent


def load_features(data, names, workers, rebuild):
    """Keep one shared feature cache directly inside the existing output folder."""
    path = ROOT / "output/model4_features.parquet"
    meta = ROOT / "output/model4_cache.json"
    if not rebuild and path.exists() and meta.exists():
        if json.loads(meta.read_text(encoding="utf-8")).get("data") == str(data):
            ridge.log(f"Using feature cache: {path}")
            return pd.read_parquet(path)
    intra = ROOT / "output/model4_intraday.parquet"
    try:
        ridge.build_intraday_cache(data, intra, pd.Timestamp("2025-12-31"), workers)
        features = ridge.build_feature_table(data, intra, path, names, pd.Timestamp("2025-12-31"))
    finally:
        intra.unlink(missing_ok=True)
    meta.write_text(json.dumps({"data": str(data)}, indent=2), encoding="utf-8")
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp500", type=Path, default=ROOT / "sp500")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild features from raw data")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data = args.sp500.resolve()
    if not (data / "panel/daily.parquet").exists():
        parser.error(f"Missing data: {data / 'panel/daily.parquet'}")
    output = ROOT / "output" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("model4_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    ridge.log(f"Results: {output}")
    definitions = json.loads((ROOT / "utils/factor_definitions.json").read_text(encoding="utf-8"))
    names = list(definitions["factors"])
    config = dict(data=str(data), alpha=1000, alpha_grid=[1000], seed=0, workers=args.workers,
                  feature_count=len(names), scope="all historically eligible stocks", strategy="top50_drop5",
                  capital=1e6, cost_bps=[0, 5, 10], risk_free_rate=0, blas_threads=2,
                  execution="signal t close; fill t+1 adjusted open; mark t+2 adjusted open",
                  selection="Retain supplied Ridge48; no new model or hyperparameter selection",
                  evaluation="2021-2023 validation; 2024-2025 testing",
                  versions={**ridge.package_versions(), "matplotlib": matplotlib.__version__})
    features = load_features(data, names, args.workers, args.rebuild)
    features["date"] = pd.to_datetime(features.date)
    features["exit_date"] = pd.to_datetime(features.exit_date)
    run_experiment(features, names, data, output, config, "model4",
                   "48 daily and intraday factors; training-only median imputation and scaling; "
                   "Ridge alpha=1000. Parameters are inherited from the supplied model; no new hyperparameter search.")


if __name__ == "__main__":
    main()
