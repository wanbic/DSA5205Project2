"""Shared linear-model training, time splits, evaluation and report outputs."""
import json
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from utils import features as ridge
from utils.backtest import run_ledger, performance


def fit_model(features, names, cutoff, penalty="ridge", alpha=1000.):
    train = features[features.target.notna() & features.exit_date.le(cutoff)
                     & features.date.ge("2018-01-01")]
    grouped = train.groupby("date").target
    y = ((train.target - grouped.transform("mean")) / grouped.transform("std")).to_numpy()
    if not np.isfinite(y).all():
        raise ValueError("Non-finite standardized training targets")
    transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    with threadpool_limits(limits=2):
        x = np.asfortranarray(transform.fit_transform(train[names].to_numpy(dtype=float)))
        if penalty == "ridge":
            model = Ridge(alpha=alpha, fit_intercept=True, solver="auto").fit(x, y)
        elif penalty == "lasso":
            model = Lasso(alpha=alpha, fit_intercept=True, precompute=True,
                          max_iter=20000, tol=1e-6, selection="cyclic").fit(x, y)
            if model.n_iter_ >= model.max_iter:
                raise RuntimeError(f"Lasso did not converge for alpha={alpha}")
        elif penalty == "elasticnet":
            model = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=True,
                               precompute=True, max_iter=20000, tol=1e-6,
                               selection="cyclic").fit(x, y)
            if model.n_iter_ >= model.max_iter:
                raise RuntimeError(f"Elastic Net did not converge for alpha={alpha}")
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
    metadata = dict(training_start="2018-01-01", label_exit_cutoff=cutoff,
                    rows=len(train), dates=int(train.date.nunique()),
                    last_signal=str(train.date.max().date()))
    return dict(model=model, transform=transform, feature_names=names), metadata


def market_inputs(source, signal_dates, assets):
    """Align signal t with execution t+1 and valuation t+2 on SPY's calendar."""
    full = pd.DatetimeIndex(sorted(source.loc[source.is_benchmark & source.ticker.eq("SPY"), "date"].unique()))
    indices = full.get_indexer(signal_dates)
    if (indices < 0).any() or not np.all(np.diff(indices) == 1):
        raise ValueError("Signal dates must be consecutive trading sessions")
    calendar = full[indices[0] + 1:indices[-1] + 3]
    if len(calendar) != len(signal_dates) + 1:
        raise ValueError("Insufficient execution/valuation dates")
    def prices(column, dates):
        return ridge.panel(source, dates, assets, column).to_numpy(dtype=float)
    dollars = source.assign(dollars=source.vwap * source.volume)
    adv = ridge.panel(dollars, full, assets, "dollars").rolling(20, min_periods=20).mean()
    return dict(open=prices("adj_open", calendar), close=prices("adj_close", calendar),
                terminal=prices("delist_ret", calendar), calendar=calendar,
                signal_dates=signal_dates, assets=assets, spy_index=assets.index("SPY"),
                adv=adv.reindex(signal_dates).to_numpy(),
                initial_marks=prices("adj_close", signal_dates[:1])[0], capital=1e6)


def prediction_metrics(predictions):
    rows = []
    for date, group in predictions.groupby("date", sort=True):
        valid = group[["score", "target"]].dropna()
        n = len(valid)
        correlated = n > 1 and valid.score.nunique() > 1 and valid.target.nunique() > 1
        sd = valid.target.std()
        z = (valid.target - valid.target.mean()) / sd if sd > 0 else pd.Series(np.nan, index=valid.index)
        rows.append(dict(date=date, observations=len(group), valid_labels=n,
                         coverage=n / len(group), ic=valid.score.corr(valid.target) if correlated else np.nan,
                         rank_ic=valid.score.rank().corr(valid.target.rank()) if correlated else np.nan,
                         squared_error=float(((valid.score - z) ** 2).sum()) if sd > 0 else np.nan,
                         zero_error=float((z ** 2).sum()) if sd > 0 else np.nan))
    daily = pd.DataFrame(rows)
    ic_sd = daily.ic.std()
    total = daily.zero_error.sum()
    metrics = dict(mean_ic=daily.ic.mean(), mean_rank_ic=daily.rank_ic.mean(),
                   ic_std=ic_sd, icir=daily.ic.mean() / ic_sd if ic_sd > 0 else np.nan,
                   label_coverage=daily.valid_labels.sum() / daily.observations.sum(),
                   standardized_target_r2_vs_zero=1 - daily.squared_error.sum() / total if total > 0 else np.nan)
    return daily, metrics


def save_plots(output, books, summary, model_name):
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": .25})
    periods = list(dict.fromkeys(period for period, _ in books))
    variants = [name for name in summary.model.unique() if name != "SPY"]
    palette = dict(zip([*variants, "SPY"], plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for column, period in enumerate(periods):
        for name in [*variants, "SPY"]:
            book = books[(period, name)]
            dates = pd.DatetimeIndex([book.entry_date.iloc[0], *book.exit_date])
            wealth = np.r_[1., book.end_nav.to_numpy() / 1e6]
            axes[0, column].plot(dates, wealth, label=name, color=palette[name])
            axes[1, column].plot(dates, wealth / np.maximum.accumulate(wealth) - 1, color=palette[name])
        axes[0, column].set_title(period.replace("validation_", "Validation ").replace("test_", "Test "))
        for ax in axes[:, column]:
            ax.tick_params(axis="x", rotation=30)
            ax.locator_params(axis="y", nbins=5)
        axes[0, column].legend()
    axes[0, 0].set_ylabel("Net wealth (initial = 1)")
    axes[1, 0].set_ylabel("Drawdown")
    fig.suptitle(f"{model_name.replace('_', ' ').capitalize()} vs SPY | 5 bps per side")
    fig.tight_layout(); fig.savefig(output / "performance.png"); plt.close(fig)
    main = summary[summary.cost_bps.eq(5)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = {p: p.replace("validation_", "Val ").replace("test_", "Test ") for p in periods}
    for ax, metric, title in [(axes[0, 0], "cagr", "CAGR"), (axes[0, 1], "sharpe_rf0", "Sharpe (rf = 0)")]:
        table = main.pivot(index="period", columns="model", values=metric).reindex(periods).rename(index=labels)
        table.plot.bar(ax=ax, rot=15, color=[palette[name] for name in table.columns])
        ax.set(title=title, xlabel="")
    r2 = main[main.model.isin(variants)].pivot(index="period", columns="model", values="standardized_target_r2_vs_zero").reindex(periods).rename(index=labels)
    r2.plot.bar(ax=axes[1, 0], rot=15, color=[palette[name] for name in r2.columns])
    axes[1, 0].set(title="R2: standardized target vs zero forecast", xlabel="")
    if len(variants) == 1:
        for period, part in summary[summary.model.eq(variants[0])].groupby("period"):
            axes[1, 1].plot(part.cost_bps, part.cagr, marker="o", label=labels[period])
    else:
        for name, part in summary[summary.period.eq("test_2024_2025") & summary.model.isin(variants)].groupby("model"):
            axes[1, 1].plot(part.cost_bps, part.cagr, marker="o", label=name, color=palette[name])
    axes[1, 1].set(title="Transaction cost sensitivity", xlabel="Cost per side (bps)", ylabel="CAGR")
    if len(variants) > 1:
        axes[1, 1].set_title("Test-period cost sensitivity")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "diagnostics.png"); plt.close(fig)


def write_notes(output, summary, model_name, description):
    columns = ["period", "model", "cagr", "annualized_volatility", "sharpe_rf0", "max_drawdown", "mean_rank_ic"]
    if "active_features" in summary:
        columns.append("active_features")
    table = summary[summary.cost_bps.eq(5)][columns]
    (output / "report_notes.md").write_text(
        f"# {model_name.capitalize()} results\n\n{description}\n\n"
        "Expanding training starts in 2018. Validation years: 2021, 2022, 2023. "
        "Each fold uses only labels with an exit date on or before the preceding year-end. "
        "The final model trains through 2023-12-29; sample counts are recorded in run_info.json. "
        "The testing period is 2024-2025, using the model fitted through 2023.\n\n"
        "Universe: all historically eligible stocks. Signals use finalized day-t data; "
        "execution is at t+1 adjusted open and valuation at t+2 adjusted open. "
        "Top50Drop5 holds about 50 equally weighted stocks with at most five discretionary replacements daily; "
        "eligibility exits and refills are separate. Initial capital: USD 1,000,000. "
        "Costs apply to absolute traded notional on both buys and sells, including entry and liquidation. "
        "The main cost is 5 bps per side; 0/10 bps are sensitivity scenarios.\n\n"
        "Fractional adjusted units, no leverage, zero cash interest and risk-free rate. "
        "Missing entries remain cash; untradable holdings are frozen; delistings use source proxies. "
        "Each evaluation period starts with a fresh portfolio; its last two sessions allow execution and valuation.\n\n"
        "CSV returns are decimals; annualization uses 252 sessions. Scores predict standardized cross-sectional "
        "returns, not calibrated returns. R2 compares the ex-post standardized target with zero; "
        "ICIR is daily mean IC divided by sample standard deviation, without annualization.\n\n"
        "```text\n" + table.to_string(index=False) + "\n```\n\n"
        "metrics.csv combines period-level results at 0/5/10 bps and annual results at 5 bps. "
        "A blank year identifies a full-period row. predictions.parquet supports later baseline comparisons. "
        "run_info.json records training counts, settings, package versions, and successful ledger checks.\n",
        encoding="utf-8")


def run_experiment(features, names, data, output, config, model_name, description, variants=None):
    source = pd.read_parquet(data / "panel/daily.parquet", columns=[
        "date", "security_id", "ticker", "is_benchmark", "adj_open", "adj_close", "delist_ret", "vwap", "volume"],
        filters=[("date", "<=", pd.Timestamp("2025-12-31"))])
    source["date"] = pd.to_datetime(source.date)
    calendar = pd.DatetimeIndex(sorted(source.loc[source.is_benchmark & source.ticker.eq("SPY"), "date"].unique()))
    comparison = variants is not None
    variants = variants or {model_name: ("ridge", 1000.)}
    summaries, annual, prediction_frames, books, training = [], [], [], {}, []
    final_bundles = {}
    for start, end, cutoff, period in [
        ("2021-01-01", "2021-12-31", "2020-12-31", "validation_2021"),
        ("2022-01-01", "2022-12-31", "2021-12-31", "validation_2022"),
        ("2023-01-01", "2023-12-31", "2022-12-31", "validation_2023"),
        ("2024-01-01", "2025-12-31", "2023-12-29", "test_2024_2025"),
    ]:
        ridge.log(f"Training/evaluating {period}")
        # Execution and valuation must both lie within the evaluation year/period.
        signal_dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))][:-2]
        sample = features[features.date.isin(signal_dates)].copy()
        assets = sorted(set(sample.security_id) | {"SPY"})
        market = market_inputs(source, signal_dates, assets)
        market["universe"] = sample.assign(eligible=True).pivot(index="date", columns="security_id", values="eligible").reindex(index=signal_dates, columns=assets).eq(True).to_numpy()
        cases, forecast_metrics, active = [], {}, {}
        for name, (penalty, alpha) in variants.items():
            # Preprocessing and labels are restricted to this fold's training data.
            bundle, metadata = fit_model(features, names, cutoff, penalty, alpha)
            training.append(dict(period=period, **({"model": name} if comparison else {}), **metadata))
            final_bundles[name] = bundle
            with threadpool_limits(limits=2):
                sample["score"] = bundle["model"].predict(bundle["transform"].transform(sample[names].to_numpy(dtype=float)))
            sample["period"] = period
            pred = sample[["date", "security_id", "exit_date", "target", "score", "period"]].copy()
            if comparison:
                pred["model"] = name
            prediction_frames.append(pred)
            _, forecast_metrics[name] = prediction_metrics(pred)
            active[name] = int(np.sum(np.abs(bundle["model"].coef_) > 1e-12))
            scores = sample.pivot(index="date", columns="security_id", values="score").reindex(index=signal_dates, columns=assets).to_numpy()
            cases.append((name, "top50_drop5", scores))
        cases.append(("SPY", "spy_buy_hold", np.zeros_like(scores)))
        for name, style, score in cases:
            for cost in [0, 5, 10]:
                book, _, _ = run_ledger(market, score, style, cost, 0.)
                np.testing.assert_allclose(book.start_nav.iloc[1:], book.end_nav.iloc[:-1], rtol=1e-10)
                np.testing.assert_allclose(book.trade_cost, book.traded_notional * cost / 10000, atol=1e-7)
                np.testing.assert_allclose(np.prod(1 + book.net_return), book.end_nav.iloc[-1] / 1e6, rtol=1e-10)
                common = dict(period=period, model=name, cost_bps=cost)
                extra = {"active_features": active.get(name, np.nan)} if comparison else {}
                summaries.append({**common, **performance(book), **forecast_metrics.get(name, {}), **extra})
                if cost == 5:
                    books[(period, name)] = book.copy()
                    for year, part in book.groupby(book.exit_date.dt.year):
                        annual.append({**common, "year": int(year), **performance(part, float(part.start_nav.iloc[0]))})
    summary = pd.DataFrame(summaries)
    pd.concat([summary, pd.DataFrame(annual)], ignore_index=True).to_csv(output / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    joblib.dump(final_bundles if comparison else bundle, output / f"{model_name}.joblib")
    save_plots(output, books, summary, model_name)
    write_notes(output, summary, model_name, description)
    config.update(model=model_name, training_folds=training, status="completed", ledger_checks="passed", cases=len(summary))
    if comparison:
        config["estimators"] = {name: bundle["model"].get_params() for name, bundle in final_bundles.items()}
    (output / "run_info.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    ridge.log(summary[summary.cost_bps.eq(5)][["period", "model", "cagr", "sharpe_rf0", "max_drawdown"]].to_string(index=False))
    ridge.log(f"Done: {output}")
