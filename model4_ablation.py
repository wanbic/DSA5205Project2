"""Small Ridge/Lasso/Elastic Net selection ablation on Model 4's 48 factors.

Run: python model4_ablation.py
All candidates use the same data, preprocessing, horizon, portfolio and costs.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
from threadpoolctl import threadpool_limits
from model4 import load_features
from utils import features as ridge
from utils.experiment import fit_model, prediction_metrics, run_experiment

ROOT = Path(__file__).resolve().parent
GRIDS = {
    "ridge": [100., 1000., 10000.],
    "lasso": [0.0001, 0.001, 0.01],
    "elasticnet": [0.0001, 0.001, 0.01],
}
LABELS = {"ridge": "Ridge", "lasso": "Lasso", "elasticnet": "ElasticNet"}


def select_models(features, names, data):
    """Use only 2021-2023 validation; retain every trial and its fold scores."""
    validation = features[features.date.le("2023-12-29")].copy()
    source = pd.read_parquet(data / "panel/daily.parquet", columns=["date", "ticker", "is_benchmark"],
                             filters=[("date", "<=", pd.Timestamp("2023-12-29"))])
    calendar = pd.DatetimeIndex(sorted(source.loc[source.is_benchmark & source.ticker.eq("SPY"), "date"].unique()))
    variants, records = {}, []
    for penalty, grid in GRIDS.items():
        candidates = []
        for alpha in grid:
            folds = []
            for year in [2021, 2022, 2023]:
                dates = calendar[calendar.year == year][:-2]
                bundle, metadata = fit_model(validation, names, f"{year - 1}-12-31", penalty, alpha)
                sample = validation.loc[validation.date.isin(dates), ["date", "target", *names]].copy()
                with threadpool_limits(limits=2):
                    sample["score"] = bundle["model"].predict(bundle["transform"].transform(sample[names].to_numpy(dtype=float)))
                _, metrics = prediction_metrics(sample)
                folds.append(dict(year=year, r2=metrics["standardized_target_r2_vs_zero"],
                                  rank_ic=metrics["mean_rank_ic"] if np.isfinite(metrics["mean_rank_ic"]) else None,
                                  active_features=int(np.sum(np.abs(bundle["model"].coef_) > 1e-12)),
                                  training_rows=metadata["rows"]))
            score = float(np.mean([row["r2"] for row in folds]))
            if not np.isfinite(score):
                raise ValueError(f"Invalid validation R2: {penalty}, alpha={alpha}")
            candidate = dict(estimator=LABELS[penalty], alpha=alpha,
                             l1_ratio=0.5 if penalty == "elasticnet" else None,
                             mean_validation_r2=score, folds=folds)
            candidates.append(candidate)
            records.append(candidate)
            ridge.log(f"{LABELS[penalty]} alpha={alpha:g}: mean validation R2={score:.8f}")
        # Within a family, exact score ties prefer its larger alpha.
        selected = max(candidates, key=lambda row: (row["mean_validation_r2"], row["alpha"]))
        variants[LABELS[penalty]] = (penalty, selected["alpha"])
    # Exact cross-family ties prefer fewer active features, then the stated family order.
    winners = [row for row in records if variants[row["estimator"]][1] == row["alpha"]]
    order = list(LABELS.values())
    winner = max(winners, key=lambda row: (row["mean_validation_r2"],
                 -np.mean([fold["active_features"] for fold in row["folds"]]),
                 -order.index(row["estimator"])))
    return variants, records, winner["estimator"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp500", type=Path, default=ROOT / "sp500")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the shared Model 4 features")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data = args.sp500.resolve()
    if not (data / "panel/daily.parquet").exists():
        parser.error(f"Missing data: {data / 'panel/daily.parquet'}")
    output = ROOT / "output" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("model4_ablation_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    ridge.log(f"Results: {output}")
    definitions = json.loads((ROOT / "utils/factor_definitions.json").read_text(encoding="utf-8"))["factors"]
    names = list(definitions)
    features = load_features(data, names, args.workers, args.rebuild)
    features["date"] = pd.to_datetime(features.date)
    features["exit_date"] = pd.to_datetime(features.exit_date)
    variants, search, winner = select_models(features, names, data)
    ridge.log(f"Validation-selected estimator: {winner}; selected settings: {variants}")
    config = dict(data=str(data), feature_count=len(names), feature_definitions=definitions,
                  alpha_grid=GRIDS, elasticnet_l1_ratio=0.5, validation_search=search,
                  selected_parameters={name: dict(penalty=penalty, alpha=alpha) for name, (penalty, alpha) in variants.items()},
                  validation_selected_estimator=winner, candidate_count=9, search_fits=27,
                  seed=0, workers=args.workers, blas_threads=2,
                  scope="all historically eligible stocks", strategy="top50_drop5", capital=1e6,
                  cost_bps=[0, 5, 10], risk_free_rate=0,
                  execution="signal t close; fill t+1 adjusted open; mark t+2 adjusted open",
                  selection="Maximize equal-weight mean 2021-2023 standardized-target R2; within-family ties prefer larger alpha; family ties prefer fewer active features then Ridge/Lasso/ElasticNet order",
                  evaluation="2021-2023 validation; 2024-2025 testing; all three selected estimators retained",
                  versions={**ridge.package_versions(), "matplotlib": matplotlib.__version__})
    settings = "; ".join(f"{name} alpha={alpha:g}" for name, (_, alpha) in variants.items())
    run_experiment(features, names, data, output, config, "model4_ablation",
                   "Estimator-selection ablation using the same 48 factors as Model 4. "
                   "Nine predefined candidates (three per family), evaluated on three expanding validation folds, "
                   "give 27 search fits. Selection maximizes equal-weight mean standardized-target validation R2. "
                   "Elastic Net fixes l1_ratio=0.5. Exact within-family ties prefer larger alpha; family ties prefer "
                   "fewer active coefficients, then Ridge/Lasso/ElasticNet order. "
                   f"Selected settings: {settings}. Validation-selected estimator: {winner}. "
                   "All three estimators are refitted on 2018-2023 and evaluated on 2024-2025 regardless of test returns. "
                   "Reported selected-parameter validation scores participated in selection. Complete trial scores "
                   "and sparsity are in run_info.json. Alpha values across penalty families are not numerically "
                   "equivalent because loss normalizations and penalties differ. "
                   "The standalone model4.py remains the original fixed Ridge(alpha=1000) reference.",
                   variants=variants)


if __name__ == "__main__":
    main()
