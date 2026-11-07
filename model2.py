"""Daily price-volume Ridge baseline: python model2.py."""
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
FEATURES = {
    "return_1d": "adj_close[t]/adj_close[t-1]-1",
    "return_5d": "adj_close[t]/adj_close[t-5]-1",
    "return_20d": "adj_close[t]/adj_close[t-20]-1",
    "momentum_63_skip5": "adj_close[t-5]/adj_close[t-63]-1",
    "momentum_252_skip21": "adj_close[t-21]/adj_close[t-252]-1",
    "day_return": "adj_close/adj_open-1",
    "overnight_return": "adj_open/previous_adj_close-1",
    "volatility_5d": "sample_std_5(daily_close_return)",
    "volatility_20d": "sample_std_20(daily_close_return)",
    "daily_range": "(raw_high-raw_low)/raw_open",
    "close_location": "(2*raw_close-raw_high-raw_low)/(raw_high-raw_low)",
    "ma_deviation_5d": "adj_close/mean_5(adj_close)-1",
    "ma_deviation_20d": "adj_close/mean_20(adj_close)-1",
    "relative_volume_5d": "adj_volume/mean_5(adj_volume).shift(1)",
    "relative_volume_20d": "adj_volume/mean_20(adj_volume).shift(1)",
    "log_dollar_volume_20d": "log(mean_20(raw_close*raw_volume))",
}


def build_features(data, end="2025-12-31"):
    """Use daily OHLCV predictors and the same eligibility/labels as Model 4."""
    columns = ["date", "security_id", "ticker", "is_benchmark", "open", "high", "low",
               "close", "vwap", "volume", "adj_open", "adj_close", "adj_volume",
               "adj_factor", "adj_warning", "in_index", "delist_ret"]
    source = pd.read_parquet(data / "panel/daily.parquet", columns=columns,
                             filters=[("date", "<=", pd.Timestamp(end))])
    source["date"] = pd.to_datetime(source.date)
    if source.duplicated(["date", "security_id"]).any():
        raise ValueError("Duplicate daily panel rows")
    calendar = pd.DatetimeIndex(sorted(source.loc[source.is_benchmark & source.ticker.eq("SPY"), "date"].unique()), name="date")
    stocks = source[~source.is_benchmark]
    assets = sorted(stocks.security_id.unique())
    p = {name: ridge.panel(stocks, calendar, assets, name) for name in columns[4:]}
    universe, _ = ridge.eligible_universe(p, calendar)
    c, o, v = p["adj_close"], p["adj_open"], p["adj_volume"]
    r = c.div(c.shift(1)).sub(1)
    bank = {
        "return_1d": r,
        "return_5d": c.div(c.shift(5)).sub(1),
        "return_20d": c.div(c.shift(20)).sub(1),
        "momentum_63_skip5": c.shift(5).div(c.shift(63)).sub(1),
        "momentum_252_skip21": c.shift(21).div(c.shift(252)).sub(1),
        "day_return": c.div(o).sub(1),
        "overnight_return": o.div(c.shift(1)).sub(1),
        "volatility_5d": r.rolling(5, min_periods=5).std(),
        "volatility_20d": r.rolling(20, min_periods=20).std(),
        "daily_range": (p["high"] - p["low"]).div(p["open"]),
        "close_location": (2 * p["close"] - p["high"] - p["low"]).div(p["high"] - p["low"]),
        "ma_deviation_5d": c.div(c.rolling(5, min_periods=5).mean()).sub(1),
        "ma_deviation_20d": c.div(c.rolling(20, min_periods=20).mean()).sub(1),
        "relative_volume_5d": v.div(v.rolling(5, min_periods=5).mean().shift(1)),
        "relative_volume_20d": v.div(v.rolling(20, min_periods=20).mean().shift(1)),
        "log_dollar_volume_20d": np.log((p["close"] * p["volume"]).rolling(20, min_periods=20).mean()),
    }
    target = ridge.build_labels(o, c, p["delist_ret"])
    ii, jj = np.where(universe.to_numpy())
    exit_dates = pd.Series(calendar, index=calendar).shift(-2).to_numpy()
    payload = dict(date=calendar.to_numpy()[ii], security_id=np.asarray(assets)[jj],
                   exit_date=exit_dates[ii], target=target.to_numpy()[ii, jj])
    for name in FEATURES:
        payload[name] = bank[name].replace([np.inf, -np.inf], np.nan).to_numpy()[ii, jj].astype("float32")
    return pd.DataFrame(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp500", type=Path, default=ROOT / "sp500")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild daily features")
    args = parser.parse_args()
    data = args.sp500.resolve()
    if not (data / "panel/daily.parquet").exists():
        parser.error(f"Missing data: {data / 'panel/daily.parquet'}")
    output = ROOT / "output" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("model2_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    ridge.log(f"Results: {output}")
    cache = ROOT / "output/model2_features.parquet"
    meta = ROOT / "output/model2_cache.json"
    cache_info = dict(data=str(data), features=FEATURES)
    if not args.rebuild and cache.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == cache_info:
        features = pd.read_parquet(cache)
    else:
        features = build_features(data)
        features.to_parquet(cache, index=False)
        meta.write_text(json.dumps(cache_info, indent=2), encoding="utf-8")
    config = dict(data=str(data), alpha=1000, alpha_grid=[1000], seed=0,
                  feature_count=len(FEATURES), feature_definitions=FEATURES,
                  predictors="daily OHLCV only; no intraday bars, residuals or interaction terms",
                  scope="all historically eligible stocks", strategy="top50_drop5", capital=1e6,
                  cost_bps=[0, 5, 10], risk_free_rate=0, blas_threads=2,
                  execution="signal t close; fill t+1 adjusted open; mark t+2 adjusted open",
                  selection="16 price-volume features and alpha=1000 fixed before evaluation; no search",
                  evaluation="2021-2023 validation; 2024-2025 testing",
                  versions={**ridge.package_versions(), "matplotlib": matplotlib.__version__})
    run_experiment(features, list(FEATURES), data, output, config, "model2",
                   "Daily price-volume Ridge baseline with 16 predefined features capturing short-term returns, "
                   "momentum, volatility, price location and trading activity. No 5-minute bars, market residuals, "
                   "Alpha101 formulas or interaction terms. Training-only median imputation and scaling; "
                   "Ridge alpha=1000, fixed to match Model 4 without hyperparameter search. "
                   "VWAP and historical membership are used only for the shared eligibility filter, not as predictors.")


if __name__ == "__main__":
    main()
