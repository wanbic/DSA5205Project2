"""High-frequency feature Ridge baseline: python model1.py."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
from utils import features as ridge
from utils.experiment import run_experiment

ROOT = Path(__file__).resolve().parent
NAMES = ["intraday_" + name for name in ridge.INTRADAY_NAMES]


def build_features(data, workers):
    """Aggregate 5-minute bars; daily data supplies eligibility, labels and mapping."""
    columns = ["date", "security_id", "ticker", "is_benchmark", "open", "high", "low",
               "close", "vwap", "volume", "adj_open", "adj_close", "adj_factor",
               "adj_warning", "in_index", "delist_ret"]
    source = pd.read_parquet(data / "panel/daily.parquet", columns=columns,
                             filters=[("date", "<=", pd.Timestamp("2025-12-31"))])
    source["date"] = pd.to_datetime(source.date)
    if source.duplicated(["date", "security_id"]).any():
        raise ValueError("Duplicate daily panel rows")
    calendar = pd.DatetimeIndex(sorted(source.loc[source.is_benchmark & source.ticker.eq("SPY"), "date"].unique()), name="date")
    stocks = source[~source.is_benchmark]
    assets = sorted(stocks.security_id.unique())
    p = {name: ridge.panel(stocks, calendar, assets, name) for name in columns[4:]}
    universe, _ = ridge.eligible_universe(p, calendar)
    target = ridge.build_labels(p["adj_open"], p["adj_close"], p["delist_ret"])
    ii, jj = np.where(universe.to_numpy())
    exits = pd.Series(calendar, index=calendar).shift(-2).to_numpy()
    result = pd.DataFrame(dict(date=calendar.to_numpy()[ii], security_id=np.asarray(assets)[jj],
                               exit_date=exits[ii], target=target.to_numpy()[ii, jj]))
    temporary = ROOT / "output/model1_intraday.parquet"
    try:
        intra = ridge.build_intraday_cache(data, temporary, pd.Timestamp("2025-12-31"), workers)
    finally:
        temporary.unlink(missing_ok=True)
    intra = intra.merge(stocks[["date", "ticker", "security_id"]],
                        on=["date", "ticker"], validate="one_to_one")
    intra = intra.rename(columns={name: "intraday_" + name for name in ridge.INTRADAY_NAMES})
    # Keep every eligible stock-day, including days with unavailable intraday data.
    result = result.merge(intra[["date", "security_id", *NAMES]], on=["date", "security_id"],
                          how="left", validate="one_to_one", sort=False)
    result[NAMES] = result[NAMES].astype("float32")
    return result


def load_features(data, workers, rebuild):
    path = ROOT / "output/model1_features.parquet"
    meta = ROOT / "output/model1_cache.json"
    info = dict(data=str(data), features=NAMES, source="5-minute regular-session aggregates")
    if not rebuild and path.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == info:
        return pd.read_parquet(path)
    # The Model 4 cache contains these exact aggregates; other features are excluded.
    shared = ROOT / "output/model4_features.parquet"
    shared_meta = ROOT / "output/model4_cache.json"
    if not rebuild and shared.exists() and shared_meta.exists() and json.loads(shared_meta.read_text(encoding="utf-8")).get("data") == str(data):
        ridge.log("Reusing the 18 intraday columns from the Model 4 feature cache")
        features = pd.read_parquet(shared, columns=["date", "security_id", "exit_date", "target", *NAMES])
    else:
        features = build_features(data, workers)
    features.to_parquet(path, index=False)
    meta.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp500", type=Path, default=ROOT / "sp500")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild directly from 5-minute bars")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data = args.sp500.resolve()
    if not (data / "panel/daily.parquet").exists():
        parser.error(f"Missing data: {data / 'panel/daily.parquet'}")
    output = ROOT / "output" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("model1_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    ridge.log(f"Results: {output}")
    features = load_features(data, args.workers, args.rebuild)
    definitions = json.loads((ROOT / "utils/factor_definitions.json").read_text(encoding="utf-8"))["factors"]
    config = dict(data=str(data), alpha=1000, alpha_grid=[1000], seed=0, workers=args.workers,
                  feature_count=len(NAMES), feature_definitions={name: definitions[name] for name in NAMES},
                  predictors="18 regular-session 5-minute aggregates only; no daily predictors or interactions",
                  scope="all historically eligible stocks", strategy="top50_drop5", capital=1e6,
                  cost_bps=[0, 5, 10], risk_free_rate=0, blas_threads=2,
                  execution="signal t close; fill t+1 adjusted open; mark t+2 adjusted open",
                  selection="18 intraday features and alpha=1000 fixed before evaluation; no search",
                  evaluation="2021-2023 validation; 2024-2025 testing",
                  versions={**ridge.package_versions(), "matplotlib": matplotlib.__version__})
    run_experiment(features, NAMES, data, output, config, "model1",
                   "High-frequency feature Ridge baseline with 18 predictors from regular-session 5-minute bars: "
                   "opening/closing returns, volume shares, realized volatility, downside variation, skewness, "
                   "price path efficiency and price-volume relationships. Valid opening/closing bars and at least "
                   "95% session coverage are required. Missing sessions use training-only median imputation; "
                   "scaling is fitted on training data. Ridge alpha=1000 is fixed. Daily data supplies only "
                   "eligibility, labels and execution prices. This uses high-frequency inputs with daily "
                   "prediction and trading, not intraday execution.")


if __name__ == "__main__":
    main()
