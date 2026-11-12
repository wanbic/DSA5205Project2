# DSA5205 Project 2

## Reproduction

### 1. Download and extract the prepared data

Under the group's collection setup, Alpaca's API rate limits make downloading the S&P 500 data take approximately 7-11 hours and Russell 2000 data approximately 13 hours. Therefore, we provide the prepared S&P 500 dataset for download so the reported experiments can be reproduced without repeating the lengthy data acquisition. The model scripts run locally without an Alpaca API key.

Download the prepared `sp500` dataset from the [shared SharePoint folder](https://nusu-my.sharepoint.com/shared?listurl=https%3A%2F%2Fnusu%2Dmy%2Esharepoint%2Ecom%2Fpersonal%2Fliyanzhao%5Fu%5Fnus%5Fedu%2FDocuments&id=%2Fpersonal%2Fliyanzhao%5Fu%5Fnus%5Fedu%2FDocuments%2Fdata%5Fshare&shareLink=1&ga=1). Extract the archive into the project root, alongside `model1.py`, so these paths exist:

```text
sp500/
  panel/daily.parquet
  raw/alpaca/calendar.csv
  raw/alpaca/5min/*.parquet
```

### 2. Install dependencies and run the experiments

Use Python 3.10+ and run the following commands from the project root. Run Model 4 first so Model 1 can reuse its intraday feature cache:

```powershell
python -m pip install -r requirements.txt
python model4.py
python model1.py
python model2.py
python model3.py
python model4_ablation.py
```

With the existing local environment:

```powershell
conda run -n dsa5205proj2 python model1.py
conda run -n dsa5205proj2 python model2.py
conda run -n dsa5205proj2 python model3.py
conda run -n dsa5205proj2 python model4.py
conda run -n dsa5205proj2 python model4_ablation.py
```

### 3. Check the results

Each main model creates `output/modelN_YYYYMMDD_HHMMSS_microseconds/` (N = 1, 2, 3, or 4); the separate ablation creates `output/model4_ablation_YYYYMMDD_HHMMSS_microseconds/`. Both use Beijing time and contain seven files:

| File | Purpose |
| --- | --- |
| `modelN.joblib` / `model4_ablation.joblib` | Final fitted model and preprocessing; the ablation contains Ridge, Lasso, and ElasticNet bundles |
| `metrics.csv` | Period results at 0/5/10 bps and annual results at 5 bps; the ablation identifies estimator families and nonzero coefficient counts |
| `predictions.parquet` | Stock-level scores and labels; the ablation includes a model column for its three estimators |
| `performance.png` | Net wealth and drawdown against SPY across all four periods |
| `diagnostics.png` | Period CAGR, Sharpe, standardized-target R2, and cost sensitivity |
| `run_info.json` | Settings, versions, training counts, completion status, and ledger checks |
| `report_notes.md` | Evaluation assumptions and main results in English |

In `metrics.csv`, a blank `year` identifies a full-period row. Returns are decimals. A completed run has `"status": "completed"` in `run_info.json`.

Each model keeps one shared feature cache directly in `output/modelN_features.parquet`, with its data path in `output/modelN_cache.json` (N = 1, 2, 3, or 4). Intraday intermediates are temporary. Cached runs still refit and evaluate the model. After changing source data, rebuild with:

```powershell
python model1.py --rebuild
python model2.py --rebuild
python model3.py --rebuild
python model4.py --rebuild
```

All scripts accept `--sp500 sp500` and `--rebuild`. Models 1 and 4 and the ablation also accept `--workers 4`. The ablation reuses Model 4's feature cache and creates no separate feature cache.

Models 2 and 3 need only `sp500/panel/daily.parquet`. Models 1 and 4 additionally use `sp500/raw/alpaca/calendar.csv` and `sp500/raw/alpaca/5min/*.parquet`. Model 1 can reuse the identical 18 intraday columns from an existing Model 4 feature cache for the same data path. With no compatible cache, or with `--rebuild`, it aggregates raw 5-minute bars independently. Models 2 and 3 build directly from daily data.

All runtime dependencies from the supplied team model have been migrated into `utils`:

- `features.py`: regular-session intraday aggregation, stock eligibility, 48 factors, and labels.
- `backtest.py`: portfolio selection, execution, cash ledger, and performance metrics.
- `experiment.py`: shared Ridge/Lasso/Elastic Net fitting, time splits, evaluation, and the seven output files.
- `factor_definitions.json`: factor descriptions and feature order.

Both cached runs and full feature rebuilds use the selected model script, `utils`, and the local `sp500` dataset. All runtime dependencies are included in this repository.

Model 4 uses 48 daily/intraday features, training-only median imputation and standardization, and Ridge(alpha=1000). Validation expands from 2018 through the preceding year-end for 2021, 2022, and 2023. Only labels whose exit date is known by the training cutoff enter training. The final model trains through 2023-12-29. Parameters are inherited from the supplied model; no new tuning is performed. The testing period is 2024-2025, using the final model fitted on 2018-2023 data.

Trading uses all historically eligible stocks and Top50Drop5: about 50 equal-weight holdings with at most five discretionary replacements per day; eligibility exits and refills are separate. Signals use day-t finalized information, fills occur at t+1 adjusted open, and valuation at t+2 adjusted open. Initial capital is USD 1,000,000. Main transaction costs are 5 bps per side on actual traded notional, including initial entry and final liquidation, with 0/10 bps sensitivity. Fractional adjusted units are allowed; leverage, cash interest, and Sharpe's risk-free rate are zero. Missing entries remain cash, untradable holdings are frozen, and delistings use source proxies. Each period starts a fresh portfolio; its final two sessions are reserved for execution and valuation.

Scores represent standardized cross-sectional returns, not calibrated return forecasts. R2 compares ex-post standardized targets with zero; ICIR is not annualized. Models 1, 2, and 3 use the same dates, labels, eligibility, preprocessing rules, strategy, and costs as Model 4. All Ridge branches retain alpha=1000. Model 2 is Ridge-only. The separate Model 4 ablation compares three regularization methods on the same 48 factors. Other baselines should follow the same protocol.

Load the saved model:

```python
import joblib
bundle = joblib.load("output/model4_TIMESTAMP/model4.joblib")
scores = bundle["model"].predict(
    bundle["transform"].transform(features[bundle["feature_names"]].to_numpy(dtype=float))
)
```


Model 2 is a Ridge-only daily price-volume baseline with 16 fixed features and alpha=1000. Short-term returns allow reversal effects, longer horizons capture momentum, and volume/range/volatility measure activity and risk. There is no hyperparameter search in this baseline. Its fitted bundle and result schema match Models 1, 3, and 4.

| Feature group | Definitions |
| --- | --- |
| Returns and momentum (5) | Adjusted close returns over 1/5/20 sessions; momentum over 63 sessions skipping 5 and 252 sessions skipping 21 |
| Return decomposition (2) | Adjusted close/open minus one; adjusted open/previous adjusted close minus one |
| Volatility (2) | Sample standard deviation of daily close returns over 5/20 sessions |
| Price location (2) | (High-low)/open; (2*close-high-low)/(high-low) |
| Moving averages (2) | Adjusted close divided by its trailing 5/20-session mean, minus one |
| Volume and liquidity (3) | Adjusted volume divided by the lagged trailing 5/20-session mean; log trailing 20-session mean of raw close times raw volume |

All predictors use data through the signal close. Only the common eligibility filter uses daily VWAP and historical membership; these are not Model 2 predictors. The baseline does not use 5-minute bars, market residuals, Alpha101 formulas, or interactions. Infinities become missing values, and training-only medians fill them. Feature formulas are included in `model2.py` and each run's `run_info.json`.


Model 1 is a high-frequency feature Ridge baseline using 18 predictors aggregated from regular-session 5-minute bars. The response and trading frequency remain daily, exactly as in Models 2, 3, and 4. This isolates the input feature set rather than changing the prediction horizon or execution rules.

| Intraday feature group | Features |
| --- | --- |
| Opening/closing returns (5) | First 30/60-minute returns, last 30/60-minute returns, full-session return |
| Volume timing (2) | First/last 30-minute volume shares |
| Variation and shape (4) | Realized volatility, downside variation share, skewness, path efficiency |
| Price location (3) | VWAP reversal, normalized session range, close-location reversal |
| Price-volume relationships (4) | Volume-weighted bar return, bar-sign volume proxy, largest-bar share, return/log-volume correlation |

Regular-session boundaries follow the exchange calendar, including half days and US daylight saving. Features require at least 95% bar coverage plus valid opening and closing bars; invalid sessions remain missing and use training-only imputation. All historically eligible stocks remain in the sample even when intraday observations are missing. Daily data is used only for eligibility, labels, security mapping and execution prices. Model 1 has no daily-return predictors, market residuals, Alpha101 formulas or interactions. Ridge alpha=1000 is fixed for all four main models. The separate ablation uses validation-selected parameters.


## Model 3: enriched daily-only control

```powershell
python model3.py
```

Model 3 uses the exact 28 Model 4 factors that can be constructed from the daily panel alone. It excludes all 18 `intraday_*` factors plus `tail_reversal_risk_scaled` and `tail_reversal_x_activity`. Its feature order follows `utils/factor_definitions.json`. Daily factor formulas are shared with Model 4 through `utils.features.make_daily_bank`; Model 3 can rebuild independently without 5-minute files or a Model 4 cache.

| Main configuration | Inputs | Feature count | Estimator |
| --- | --- | --- | --- |
| Model 1 | Intraday aggregates only | 18 | Ridge(alpha=1000) |
| Model 2 | Basic daily price-volume indicators | 16 | Ridge(alpha=1000) |
| Model 3 | Enriched daily factors from Model 4 | 28 | Ridge(alpha=1000) |
| Model 4 | Model 3 plus intraday aggregates and tail interactions | 48 | Ridge(alpha=1000) |

The 28 daily factors comprise reversals, momentum, day/night return decomposition, market-residual reversals, risk/liquidity, price pressure, six Alpha101 formulas, and three daily interactions. They use daily stock prices/volume, daily SPY returns, and the common historical eligibility filter. There is no new hyperparameter search or choice based on test performance.

The report can progress from basic daily features (Model 2), to enriched daily signals (Model 3), to daily plus intraday information (Model 4). Model 2 and Model 3 change the feature design and are not strictly nested. The cleaner nested comparison is Model 3 versus the standalone Model 4: exactly the same 28 daily predictors are retained, and 20 intraday-dependent features are added. This measures the joint incremental contribution of 18 intraday aggregates and two tail interactions; it does not separately identify sampling-frequency and interaction effects. Model 1 provides a complementary intraday-only view. The separate estimator ablation addresses regularization rather than feature-set changes.

Model 3 uses the same training-only preprocessing, target, date splits, universe, Top50Drop5, costs, and seven output files. Its cache is `output/model3_features.parquet`, with metadata in `output/model3_cache.json`. Feature definitions and the 20 excluded Model 4 names are recorded in each `run_info.json`.


## Model 4 estimator-selection ablation

```powershell
python model4_ablation.py
```

The ablation keeps Model 4's 48 factors, sample, label, preprocessing, Top50Drop5, and costs fixed. Only the estimator and regularization settings change. The standalone `model4.py` remains fixed Ridge(alpha=1000).

| Estimator | Alpha candidates | Other settings |
| --- | --- | --- |
| Ridge | 100, 1000, 10000 | Automatic solver |
| Lasso | 0.0001, 0.001, 0.01 | Cyclic coordinate descent |
| Elastic Net | 0.0001, 0.001, 0.01 | Fixed l1_ratio=0.5; cyclic coordinate descent |

There are nine candidate configurations and 27 search fits across the 2021/2022/2023 expanding validation folds. Candidate selection maximizes equal-weight mean standardized-target R2. Within a family, exact ties prefer larger alpha. Across families, exact ties prefer fewer average active features, then Ridge/Lasso/ElasticNet order. All selection reads only data through 2023. Selected-parameter validation results participated in selection and are not a fresh test.

Each family's selected configuration is refitted on mature 2018-2023 labels and evaluated on 2024-2025; all three test results are retained regardless of returns. The validation-selected family is recorded in `run_info.json`, together with every candidate's fold scores, active-feature counts, selected parameters, and final estimator settings. Ridge's raw alpha is not directly comparable with Lasso/Elastic Net alpha because the objectives use different loss normalizations. This small grid is a controlled ablation, not an exhaustive estimator benchmark.

The seven-file layout remains unchanged. The ablation's `metrics.csv` and figures compare Ridge, Lasso, ElasticNet, and SPY. `active_features` counts coefficients with absolute value greater than 1e-12. `predictions.parquet` identifies each estimator in its `model` column. Load the three final bundles as follows:

```python
comparison = joblib.load("output/model4_ablation_TIMESTAMP/model4_ablation.joblib")
ridge_bundle = comparison["Ridge"]
lasso_bundle = comparison["Lasso"]
elasticnet_bundle = comparison["ElasticNet"]
```

Each bundle has `model`, `transform`, and `feature_names` fields. Coefficients are in `model.coef_`, in `feature_names` order. `model2.py` compares Ridge with SPY.


## Rebuild the report

`Group_6_Report.tex` contains the English report and `Group_6_Report.pdf` is the compiled report. The report includes data sources, coverage, cleaning rules, the team roster, and member contributions. Collection statistics and cleaning rules are checked against the local files and model code. Tables are generated from completed experiment outputs, and figures use reconstructed ledgers checked against their recorded metrics.

```powershell
python utils/build_report_assets.py --latest
pdflatex -interaction=nonstopmode -halt-on-error Group_6_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error Group_6_Report.tex
```

Run these commands after all five experiments complete. PDF compilation requires a LaTeX distribution such as MiKTeX or TeX Live with `pdflatex` available on the command path. Use the project Python environment for the asset builder. The asset builder creates `report_assets/tables.tex` and six combined figures. Its default pins the report's original run IDs; `--latest` finds each configuration's newest completed run after a fresh reproduction. Keep `report_assets` with `Group_6_Report.tex`. Current paths and software versions appear in the report. Figures and tables should be regenerated together if an experiment changes.
