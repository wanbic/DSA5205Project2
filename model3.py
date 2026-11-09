"""Enriched daily-only Ridge baseline: python model3.py.

Use the exact 28 daily-only factors from Model 4, excluding every intraday input.
"""
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
INTRADAY_INTERACTIONS = {"tail_reversal_risk_scaled", "tail_reversal_x_activity"}


def build_features(data, names, end="2025-12-31"):
    """Rebuild from the daily panel alone; preserve Model 4's rows and formulas."""
    columns = ["date", "security_id", "ticker", "is_benchmark", "open", "high", "low",
               "close", "vwap", "volume", "adj_open", "adj_close", "adj_volume",
               "adj_factor", "adj_warning", "in_index", "delist_ret"]
    source = pd.read_parquet(data / "panel/daily.parquet", columns=columns,
                             filters=[("date", "<=", pd.Timestamp(end))])
    source["date"] = pd.to_datetime(source.date)
    if source.duplicated(["date", "security_id"]).any():
        raise ValueError("Duplicate daily panel rows")
    benchmark = source[source.is_benchmark & source.ticker.eq("SPY")]
    calendar = pd.DatetimeIndex(sorted(benchmark.date.unique()), name="date")
    stocks = source[~source.is_benchmark]
    assets = sorted(stocks.security_id.unique())
    p = {name: ridge.panel(stocks, calendar, assets, name) for name in columns[4:]}
    universe, _ = ridge.eligible_universe(p, calendar)
    spy = benchmark.set_index("date").adj_close.sort_index().reindex(calendar)
    bank = ridge.make_daily_bank(p, universe, spy.pct_change(fill_method=None))
    if set(bank) != set(names):
        raise ValueError("Daily factor bank does not match the 28 selected features")
    target = ridge.build_labels(p["adj_open"], p["adj_close"], p["delist_ret"])
    ii, jj = np.where(universe.to_numpy())
    exits = pd.Series(calendar, index=calendar).shift(-2).to_numpy()
    payload = dict(date=calendar.to_numpy()[ii], security_id=np.asarray(assets)[jj],
                   exit_date=exits[ii], target=target.to_numpy()[ii, jj])
    for name in names:
        values = bank[name].to_numpy()[ii, jj].astype("float32")
        values[~np.isfinite(values)] = np.nan
        payload[name] = values
    return pd.DataFrame(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp500", type=Path, default=ROOT / "sp500")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the 28 factors from daily data")
    args = parser.parse_args()
    data = args.sp500.resolve()
    if not (data / "panel/daily.parquet").exists():
        parser.error(f"Missing data: {data / 'panel/daily.parquet'}")
    output = ROOT / "output" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("model3_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    ridge.log(f"Results: {output}")
    definitions = json.loads((ROOT / "utils/factor_definitions.json").read_text(encoding="utf-8"))["factors"]
    names = [name for name in definitions if not name.startswith("intraday_") and name not in INTRADAY_INTERACTIONS]
    excluded = [name for name in definitions if name not in names]
    if len(names) != 28 or len(excluded) != 20:
        raise ValueError("Expected 28 daily factors and 20 excluded intraday-dependent factors")
    cache = ROOT / "output/model3_features.parquet"
    meta = ROOT / "output/model3_cache.json"
    cache_info = dict(data=str(data), features=names, source="daily panel only")
    if not args.rebuild and cache.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == cache_info:
        features = pd.read_parquet(cache)
    else:
        features = build_features(data, names)
        features.to_parquet(cache, index=False)
        meta.write_text(json.dumps(cache_info, indent=2), encoding="utf-8")
    config = dict(data=str(data), alpha=1000, alpha_grid=[1000], seed=0,
                  feature_count=len(names), feature_definitions={name: definitions[name] for name in names},
                  excluded_model4_features=excluded,
                  predictors="28 enriched daily factors; exact daily-only subset of Model 4",
                  scope="all historically eligible stocks", strategy="top50_drop5", capital=1e6,
                  cost_bps=[0, 5, 10], risk_free_rate=0, blas_threads=2,
                  execution="signal t close; fill t+1 adjusted open; mark t+2 adjusted open",
                  selection="28 daily-only features and alpha=1000 fixed before evaluation; no search",
                  evaluation="2021-2023 validation; 2024-2025 testing",
                  versions={**ridge.package_versions(), "matplotlib": matplotlib.__version__})
    run_experiment(features, names, data, output, config, "model3",
                   "Enriched daily-only Ridge baseline using the exact 28 factors in Model 4 that do not "
                   "require 5-minute data: daily reversals, momentum, return decomposition, market-residual "
                   "reversals, risk/liquidity, price pressure, six Alpha101 formulas and three daily interactions. "
                   "All 18 intraday aggregates and both tail-return interactions are excluded. "
                   "Training-only median imputation and scaling; Ridge alpha=1000. No hyperparameter search. "
                   "Model 2 versus Model 3 compares basic and enriched daily feature designs; these sets are "
                   "not strictly nested. Model 3 versus fixed-alpha Model 4 is a nested input ablation: "
                   "it adds 18 intraday aggregates and two intraday-dependent interactions, with everything "
                   "else held constant. This estimates the incremental value of that 20-feature block, "
                   "not the isolated effect of sampling frequency or the two interactions separately.")


if __name__ == "__main__":
    main()
