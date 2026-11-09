"""Ridge48 feature construction from local daily and 5-minute market data.

Migrated from the supplied team model; original calculation functions preserved.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INTRADAY_NAMES = [
    "head30",
    "head60",
    "tail30",
    "tail60",
    "head_volume_share",
    "tail_volume_share",
    "rv",
    "downside_share",
    "skew",
    "efficiency",
    "vwap_reversal",
    "range",
    "close_location_reversal",
    "weighted_bar_return",
    "signed_bar_volume",
    "largest_bar_share",
    "return_volume_corr",
    "rth_return",
]


def aggregate_bars(bars, calendar):
    if bars.empty:
        return pd.DataFrame()
    b = bars.sort_values("t").drop_duplicates("t", keep="last").copy()
    local = b.t.dt.tz_convert("America/New_York")
    b["date"] = local.dt.tz_localize(None).dt.normalize()
    b["minute"] = local.dt.hour * 60 + local.dt.minute
    cal = calendar.set_index("date")
    b["session_open"] = b.date.map(cal.open_minute)
    b["session_close"] = b.date.map(cal.close_minute)
    valid = (b.minute.ge(b.session_open) & b.minute.lt(b.session_close)
             & b.minute.mod(5).eq(0) & b[["o", "h", "l", "c", "vw"]].gt(0).all(axis=1)
             & b.v.gt(0) & b.h.ge(b[["o", "c", "l"]].max(axis=1))
             & b.l.le(b[["o", "c", "h"]].min(axis=1)))
    b = b.loc[valid].copy()
    if b.empty:
        return pd.DataFrame()
    g = b.groupby("date", sort=True)
    previous = g.c.shift(1).fillna(b.o)
    b["r"] = np.log(b.c / previous)
    b["r2"] = b.r ** 2
    b["down2"] = b.r2.where(b.r.lt(0), 0)
    b["r3"] = b.r ** 3
    b["abs_r"] = b.r.abs()
    b["r_v"] = b.r * b.v
    b["dollars"] = b.vw * b.v
    b["signed_v"] = np.sign(b.r) * b.v
    b["logv"] = np.log(b.v)
    b["logv2"] = b.logv ** 2
    b["r_logv"] = b.r * b.logv
    out = b.groupby("date").agg(
        bars=("t", "size"), first_minute=("minute", "min"), last_minute=("minute", "max"),
        session_open=("session_open", "first"), session_close=("session_close", "first"),
        first_open=("o", "first"), last_close=("c", "last"), high=("h", "max"), low=("l", "min"),
        volume=("v", "sum"), dollars=("dollars", "sum"), sum_r=("r", "sum"),
        sum_r2=("r2", "sum"), sum_down2=("down2", "sum"), sum_r3=("r3", "sum"),
        sum_abs_r=("abs_r", "sum"), sum_r_v=("r_v", "sum"), sum_signed_v=("signed_v", "sum"),
        sum_logv=("logv", "sum"), sum_logv2=("logv2", "sum"), sum_r_logv=("r_logv", "sum"),
        max_abs_bar=("abs_r", "max"))
    def pick(mask, column):
        return b.loc[mask].set_index("date")[column].reindex(out.index)
    opening = pick(b.minute.eq(b.session_open), "o")
    closing = pick(b.minute.eq(b.session_close - 5), "c")
    out["head30"] = pick(b.minute.eq(b.session_open + 25), "c") / opening - 1
    out["head60"] = pick(b.minute.eq(b.session_open + 55), "c") / opening - 1
    out["tail30"] = closing / pick(b.minute.eq(b.session_close - 30), "o") - 1
    out["tail60"] = closing / pick(b.minute.eq(b.session_close - 60), "o") - 1
    for name, mask in [("head_volume_share", b.minute.lt(b.session_open + 30)),
                       ("tail_volume_share", b.minute.ge(b.session_close - 30))]:
        out[name] = b.loc[mask].groupby("date").v.sum() / out.volume
    out["rth_return"] = closing / opening - 1
    out["rv"] = np.sqrt(out.sum_r2)
    out["downside_share"] = out.sum_down2 / out.sum_r2
    out["skew"] = out.sum_r3 / out.sum_r2.pow(1.5)
    out["efficiency"] = np.log(closing / opening).abs() / out.sum_abs_r
    out["vwap"] = out.dollars / out.volume
    out["vwap_reversal"] = 1 - closing / out.vwap
    out["range"] = (out.high - out.low) / opening
    out["close_location_reversal"] = -(2 * closing - out.high - out.low) / (out.high - out.low)
    out["weighted_bar_return"] = out.sum_r_v / out.volume
    # This is a BAR-SIGN volume proxy, not actual buyer/seller initiated flow.
    out["signed_bar_volume"] = out.sum_signed_v / out.volume
    out["largest_bar_share"] = out.max_abs_bar / out.rv
    covariance = out.sum_r_logv - out.sum_r * out.sum_logv / out.bars
    variance_r = out.sum_r2 - out.sum_r ** 2 / out.bars
    variance_v = out.sum_logv2 - out.sum_logv ** 2 / out.bars
    out["return_volume_corr"] = covariance / np.sqrt(variance_r.clip(lower=0) * variance_v.clip(lower=0))
    out["coverage"] = out.bars / ((out.session_close - out.session_open) / 5)
    good = out.coverage.ge(.95) & opening.notna() & closing.notna()
    features = ["head30", "head60", "tail30", "tail60", "head_volume_share", "tail_volume_share", "rth_return",
                "rv", "downside_share", "skew", "efficiency", "vwap_reversal", "range",
                "close_location_reversal", "weighted_bar_return", "signed_bar_volume", "largest_bar_share", "return_volume_corr"]
    out.loc[~good, features] = np.nan
    out["valid_session"] = good
    return out[["bars", "coverage", "valid_session", "volume", "dollars"] + features].replace([np.inf, -np.inf], np.nan).reset_index()


def log(message: str) -> None:
    print(message, flush=True)


def read_one_intraday(
    path: Path, calendar: pd.DataFrame, cutoff_exclusive: pd.Timestamp
) -> tuple[pd.DataFrame, int]:
    bars = pd.read_parquet(
        path,
        columns=["t", "o", "h", "l", "c", "v", "vw"],
        filters=[("t", "<", cutoff_exclusive)],
        use_threads=False,
    )
    if not bars.empty and bars.t.max() >= cutoff_exclusive:
        raise AssertionError(f"Post-cutoff bar found in {path}")
    daily = aggregate_bars(bars, calendar)
    if not daily.empty:
        daily.insert(1, "ticker", path.stem)
    return daily, len(bars)


def build_intraday_cache(
    sp500: Path,
    destination: Path,
    training_end: pd.Timestamp,
    workers: int,
) -> pd.DataFrame:
    calendar_path = sp500 / "raw" / "alpaca" / "calendar.csv"
    bars_dir = sp500 / "raw" / "alpaca" / "5min"
    if not calendar_path.exists():
        raise FileNotFoundError(calendar_path)
    files = sorted(bars_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No 5-minute parquet files found in {bars_dir}")

    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    for column in ["open", "close"]:
        values = calendar[column].str.split(":", expand=True).astype(int)
        calendar[column + "_minute"] = values[0] * 60 + values[1]
    calendar = calendar[calendar.date.le(training_end)].copy()
    cutoff_exclusive = (training_end + pd.Timedelta(days=1)).tz_localize("UTC")

    chunks: list[pd.DataFrame] = []
    source_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                read_one_intraday, path, calendar, cutoff_exclusive
            ): path
            for path in files
        }
        for completed, future in enumerate(as_completed(futures), 1):
            frame, rows = future.result()
            source_rows += rows
            if not frame.empty:
                chunks.append(frame)
            if completed % 25 == 0 or completed == len(files):
                log(
                    f"Aggregated {completed}/{len(files)} tickers; "
                    f"read {source_rows:,} five-minute bars"
                )

    result = pd.concat(chunks, ignore_index=True).sort_values(["date", "ticker"])
    if result.duplicated(["date", "ticker"]).any():
        raise AssertionError("Duplicate date/ticker rows in intraday aggregation")
    if result.date.max() > training_end:
        raise AssertionError("Intraday cache extends beyond training cutoff")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    log(f"Saved {len(result):,} intraday daily rows to {destination}")
    return result


def panel(
    source: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    assets: list[str],
    column: str,
) -> pd.DataFrame:
    return source.pivot(index="date", columns="security_id", values=column).reindex(
        index=calendar, columns=assets
    )


def build_labels(
    open_price: pd.DataFrame,
    close_price: pd.DataFrame,
    terminal_return: pd.DataFrame,
) -> pd.DataFrame:
    """Build the audited t+1-open to t+2-open target."""
    entry = open_price.shift(-1)
    exit_price = open_price.shift(-2)
    terminal = terminal_return.shift(-1)
    terminal_label = terminal.notna() & entry.gt(0)
    target = exit_price.div(entry).sub(1)
    liquidation = close_price.shift(-1).mul(1 + terminal).div(entry).sub(1)
    target = target.mask(terminal_label, liquidation)
    within_sample = np.arange(len(open_price)) < len(open_price) - 2
    target = target.where(within_sample[:, None] & entry.gt(0))
    return target.replace([np.inf, -np.inf], np.nan)


def make_alpha101_factors(
    p: dict[str, pd.DataFrame], universe: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    rank = lambda x: x.where(universe).rank(axis=1, pct=True)
    log_volume_change = np.log(p["adj_volume"].where(p["adj_volume"].gt(0))).diff(2)
    intraday_return = p["close"].div(p["open"]).sub(1)
    factors = {
        "alpha002": -rank(log_volume_change).rolling(6, min_periods=6).corr(
            rank(intraday_return)
        ),
        "alpha003": -rank(p["open"]).rolling(10, min_periods=10).corr(
            rank(p["volume"])
        ),
        "alpha006": -p["adj_open"].rolling(10, min_periods=10).corr(
            p["adj_volume"]
        ),
        "alpha012": -np.sign(p["adj_volume"].diff()).mul(
            p["adj_close"].diff().div(p["adj_factor"])
        ),
        "alpha041": np.sqrt(p["high"].mul(p["low"])) - p["vwap"],
        "alpha101": (p["close"] - p["open"])
        / (p["high"] - p["low"] + 0.001),
    }
    return {
        name: values.replace([np.inf, -np.inf], np.nan).where(universe)
        for name, values in factors.items()
    }


def make_daily_factors(
    p: dict[str, pd.DataFrame], spy_return: pd.Series
) -> dict[str, pd.DataFrame]:
    c, o = p["adj_close"], p["adj_open"]
    r = c / c.shift(1) - 1
    day, night = c / o - 1, o / c.shift(1) - 1
    dollars = p["vwap"] * p["volume"]
    beta = (
        r.rolling(126, min_periods=100)
        .cov(spy_return)
        .div(spy_return.rolling(126, min_periods=100).var(), axis=0)
        .shift(1)
    )
    residual = r - beta.mul(spy_return, axis=0)
    volatility = r.rolling(20, min_periods=20).std()
    factors = {
        "reversal_1d": -r,
        "reversal_5d": 1 - c / c.shift(5),
        "reversal_20d": 1 - c / c.shift(20),
        "momentum_63_skip5": c.shift(5) / c.shift(63) - 1,
        "momentum_252_skip21": c.shift(21) / c.shift(252) - 1,
        "day_return_1": day,
        "night_return_1": night,
        "residual_reversal_1": -residual / volatility.shift(1),
        "residual_reversal_5": -residual.rolling(5, min_periods=5).sum()
        / (volatility.shift(1) * np.sqrt(5)),
        "volatility_20": volatility,
        "illiquidity_20": (r.abs() / dollars)
        .rolling(20, min_periods=20)
        .mean()
        * 1e6,
        "relative_dollar_volume": dollars
        / dollars.rolling(20, min_periods=20).mean().shift(1),
        "normalized_alpha041": (
            np.sqrt(p["high"] * p["low"]) - p["vwap"]
        )
        / p["close"],
        "normalized_vwap_reversal": (p["vwap"] - p["close"])
        / p["close"],
        "close_location_reversal": -(
            2 * p["close"] - p["high"] - p["low"]
        )
        / (p["high"] - p["low"]),
    }
    for length in [5, 20]:
        factors[f"day_return_{length}"] = np.expm1(
            np.log1p(day).rolling(length, min_periods=length).sum()
        )
        factors[f"night_return_{length}"] = np.expm1(
            np.log1p(night).rolling(length, min_periods=length).sum()
        )
    return {
        name: values.replace([np.inf, -np.inf], np.nan)
        for name, values in factors.items()
    }


def make_daily_bank(p, universe, spy_return):
    """The 28 Model 4 factors that require daily data only."""
    bank = make_daily_factors(p, spy_return)
    bank.update(make_alpha101_factors(p, universe))
    bank["residual_reversal_x_illiquidity"] = bank[
        "residual_reversal_1"
    ] * bank["illiquidity_20"].where(universe).rank(axis=1, pct=True)
    bank["residual_reversal_x_volume"] = bank[
        "residual_reversal_1"
    ] * bank["relative_dollar_volume"].clip(0.2, 5)
    bank["gap_reversal_after_disagreement"] = -bank["night_return_1"] * (
        np.sign(bank["night_return_1"]) != np.sign(bank["day_return_1"])
    )
    return bank


def eligible_universe(
    p: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    good = p["close"].ge(5) & p["volume"].gt(0) & p["adj_factor"].gt(0)
    good &= p["high"].ge(p["open"]) & p["high"].ge(p["close"])
    good &= p["low"].le(p["open"]) & p["low"].le(p["close"])
    warning = p["adj_warning"].eq(True).rolling(10, min_periods=1).max().eq(1)
    adv = (p["vwap"] * p["volume"]).rolling(20, min_periods=20).mean()
    good &= adv.ge(1e7)
    good &= p["close"].notna().rolling(60, min_periods=60).sum().eq(60)
    universe = p["in_index"].eq(True) & good & ~warning
    universe &= (calendar >= pd.Timestamp("2018-01-02"))[:, None]
    return universe, adv


def build_feature_table(
    sp500: Path,
    intraday_path: Path,
    destination: Path,
    factor_names: list[str],
    training_end: pd.Timestamp,
) -> pd.DataFrame:
    panel_path = sp500 / "panel" / "daily.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    columns = [
        "date",
        "security_id",
        "ticker",
        "is_benchmark",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "adj_open",
        "adj_close",
        "adj_volume",
        "adj_factor",
        "adj_warning",
        "in_index",
        "delist_ret",
    ]
    source = pd.read_parquet(
        panel_path,
        columns=columns,
        filters=[("date", "<=", training_end)],
    )
    source["date"] = pd.to_datetime(source.date)
    if source.duplicated(["date", "security_id"]).any():
        raise AssertionError("Duplicate date/security_id rows in daily panel")

    benchmark = source[source.is_benchmark & source.ticker.eq("SPY")]
    calendar = pd.DatetimeIndex(sorted(benchmark.date.unique()), name="date")
    stock_source = source[~source.is_benchmark].copy()
    assets = sorted(stock_source.security_id.unique())
    required_panels = [
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "adj_open",
        "adj_close",
        "adj_volume",
        "adj_factor",
        "adj_warning",
        "in_index",
        "delist_ret",
    ]
    p = {
        name: panel(stock_source, calendar, assets, name)
        for name in required_panels
    }
    universe, _ = eligible_universe(p, calendar)
    log(
        f"Daily panel: {len(calendar):,} sessions, {len(assets):,} assets, "
        f"{int(universe.to_numpy().sum()):,} eligible stock-days"
    )

    spy = benchmark.set_index("date").adj_close.sort_index().reindex(calendar)
    bank = make_daily_bank(p, universe, spy.pct_change(fill_method=None))

    intra = pd.read_parquet(intraday_path)
    intra["date"] = pd.to_datetime(intra.date)
    mapping = stock_source[["date", "ticker", "security_id"]]
    if mapping.duplicated(["date", "ticker"]).any():
        raise AssertionError("Daily panel has duplicate date/ticker mappings")
    intra = intra.merge(mapping, on=["date", "ticker"], validate="one_to_one")
    intrabank = {
        name: intra.pivot(index="date", columns="security_id", values=name).reindex(
            index=calendar, columns=assets
        )
        for name in INTRADAY_NAMES
    }
    bank.update({"intraday_" + name: values for name, values in intrabank.items()})

    bank["tail_reversal_risk_scaled"] = (
        -bank["intraday_tail30"] / bank["intraday_rv"]
    )
    tail_share = bank["intraday_tail_volume_share"]
    bank["tail_reversal_x_activity"] = (
        -bank["intraday_tail30"]
        * tail_share
        / tail_share.rolling(20, min_periods=15).mean().shift(1)
    )

    missing = [name for name in factor_names if name not in bank]
    extra = sorted(set(bank) - set(factor_names))
    if missing or extra:
        raise AssertionError(f"Factor bank mismatch; missing={missing}, extra={extra}")

    target = build_labels(p["adj_open"], p["adj_close"], p["delist_ret"])
    ii, jj = np.where(universe.to_numpy())
    exit_dates = pd.Series(calendar, index=calendar).shift(-2).to_numpy()
    payload: dict[str, Any] = {
        "date": calendar.to_numpy()[ii],
        "security_id": np.asarray(assets)[jj],
        "exit_date": exit_dates[ii],
        "target": target.to_numpy()[ii, jj],
    }
    for completed, name in enumerate(factor_names, 1):
        values = bank[name].to_numpy()[ii, jj].astype("float32")
        values[~np.isfinite(values)] = np.nan
        payload[name] = values
        if completed % 8 == 0 or completed == len(factor_names):
            log(f"Materialized {completed}/{len(factor_names)} factors")
    result = pd.DataFrame(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    log(f"Saved {len(result):,} feature rows to {destination}")
    return result


def package_versions() -> dict[str, str]:
    names = [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "joblib",
        "threadpoolctl",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
