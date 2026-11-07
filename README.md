# DSA5205 Project 2

Run a model from the project root with Python 3.10+:

```powershell
python -m pip install -r requirements.txt
python model1.py
python model2.py
python model4.py
```

With the existing local environment:

```powershell
conda run -n dsa5205proj2 python model1.py
conda run -n dsa5205proj2 python model2.py
conda run -n dsa5205proj2 python model4.py
```

Each run creates `output/modelN_YYYYMMDD_HHMMSS_microseconds/` (N = 1, 2, or 4) using Beijing time, with seven files:

| File | Purpose |
| --- | --- |
| `modelN.joblib` | Final Ridge model, fitted preprocessing, and feature order |
| `metrics.csv` | Period results at 0/5/10 bps and annual results at 5 bps |
| `predictions.parquet` | Stock-level scores and labels for later baseline comparisons |
| `performance.png` | Net wealth and drawdown against SPY across all four periods |
| `diagnostics.png` | Period CAGR, Sharpe, standardized-target R2, and cost sensitivity |
| `run_info.json` | Settings, versions, training counts, completion status, and ledger checks |
| `report_notes.md` | Evaluation assumptions and main results in English |

In `metrics.csv`, a blank `year` identifies a full-period row. Returns are decimals. A completed run has `"status": "completed"` in `run_info.json`.

Each model keeps one shared feature cache directly in `output/modelN_features.parquet`, with its data path in `output/modelN_cache.json` (N = 1, 2, or 4). Intraday intermediates are temporary. Cached runs still refit and evaluate the model. After changing source data, rebuild with:

```powershell
python model1.py --rebuild
python model2.py --rebuild
python model4.py --rebuild
```

All models accept `--sp500 sp500` and `--rebuild`. Models 1 and 4 also accept `--workers 4`.

Model 2 needs only `sp500/panel/daily.parquet`. Models 1 and 4 additionally use `sp500/raw/alpaca/calendar.csv` and `sp500/raw/alpaca/5min/*.parquet`. Model 1 can reuse the identical 18 intraday columns from an existing Model 4 feature cache for the same data path. With no compatible cache, or with `--rebuild`, it aggregates raw 5-minute bars independently. Model 2 builds directly from daily data.

All runtime dependencies from the supplied team model have been migrated into `utils`:

- `features.py`: regular-session intraday aggregation, stock eligibility, 48 factors, and labels.
- `backtest.py`: portfolio selection, execution, cash ledger, and performance metrics.
- `experiment.py`: shared Ridge fitting, time splits, evaluation, and the seven output files.
- `factor_definitions.json`: factor descriptions and feature order.

The original model kit is no longer required and can be deleted. Both cached runs and full raw-data rebuilds use only the selected model script, `utils`, and the local `sp500` dataset. Migrated calculation functions retain the supplied implementation; unused training commands, report exporters, and frozen-forecast replay code were excluded.

Model 4 uses 48 daily/intraday features, training-only median imputation and standardization, and Ridge(alpha=1000). Validation expands from 2018 through the preceding year-end for 2021, 2022, and 2023. Only labels whose exit date is known by the training cutoff enter training. The final model trains through 2023-12-29. Parameters are inherited from the supplied model; no new tuning is performed. The testing period is 2024-2025, using the final model fitted on 2018-2023 data.

Trading uses all historically eligible stocks and Top50Drop5: about 50 equal-weight holdings with at most five discretionary replacements per day; eligibility exits and refills are separate. Signals use day-t finalized information, fills occur at t+1 adjusted open, and valuation at t+2 adjusted open. Initial capital is USD 1,000,000. Main transaction costs are 5 bps per side on actual traded notional, including initial entry and final liquidation, with 0/10 bps sensitivity. Fractional adjusted units are allowed; leverage, cash interest, and Sharpe's risk-free rate are zero. Missing entries remain cash, untradable holdings are frozen, and delistings use source proxies. Each period starts a fresh portfolio; its final two sessions are reserved for execution and valuation.

Scores represent standardized cross-sectional returns, not calibrated return forecasts. R2 compares ex-post standardized targets with zero; ICIR is not annualized. Models 1 and 2 use exactly the same dates, labels, eligibility, preprocessing, Ridge alpha, strategy, and costs as Model 4; the feature set is the intended difference. Other baselines should follow the same protocol.

Load the saved model:

```python
import joblib
bundle = joblib.load("output/model4_TIMESTAMP/model4.joblib")
scores = bundle["model"].predict(
    bundle["transform"].transform(features[bundle["feature_names"]].to_numpy(dtype=float))
)
```


Model 2 is a fixed 16-feature daily price-volume Ridge baseline. Short-term returns allow reversal effects, longer horizons capture momentum, and volume/range/volatility measure activity and risk. Ridge stabilizes correlated indicators. Alpha=1000 matches Model 4; no parameter search or test-period model selection is performed.

| Feature group | Definitions |
| --- | --- |
| Returns and momentum (5) | Adjusted close returns over 1/5/20 sessions; momentum over 63 sessions skipping 5 and 252 sessions skipping 21 |
| Return decomposition (2) | Adjusted close/open minus one; adjusted open/previous adjusted close minus one |
| Volatility (2) | Sample standard deviation of daily close returns over 5/20 sessions |
| Price location (2) | (High-low)/open; (2*close-high-low)/(high-low) |
| Moving averages (2) | Adjusted close divided by its trailing 5/20-session mean, minus one |
| Volume and liquidity (3) | Adjusted volume divided by the lagged trailing 5/20-session mean; log trailing 20-session mean of raw close times raw volume |

All predictors use data through the signal close. Only the common eligibility filter uses daily VWAP and historical membership; these are not Model 2 predictors. The baseline does not use 5-minute bars, market residuals, Alpha101 formulas, or interactions. Infinities become missing values, and training-only medians fill them. Feature formulas are included in `model2.py` and each run's `run_info.json`.


Model 1 is a high-frequency feature Ridge baseline using 18 predictors aggregated from regular-session 5-minute bars. The response and trading frequency remain daily, exactly as in Models 2 and 4. This isolates the input feature set rather than changing the prediction horizon or execution rules.

| Intraday feature group | Features |
| --- | --- |
| Opening/closing returns (5) | First 30/60-minute returns, last 30/60-minute returns, full-session return |
| Volume timing (2) | First/last 30-minute volume shares |
| Variation and shape (4) | Realized volatility, downside variation share, skewness, path efficiency |
| Price location (3) | VWAP reversal, normalized session range, close-location reversal |
| Price-volume relationships (4) | Volume-weighted bar return, bar-sign volume proxy, largest-bar share, return/log-volume correlation |

Regular-session boundaries follow the exchange calendar, including half days and US daylight saving. Features require at least 95% bar coverage plus valid opening and closing bars; invalid sessions remain missing and use training-only imputation. All historically eligible stocks remain in the sample even when intraday observations are missing. Daily data is used only for eligibility, labels, security mapping and execution prices. Model 1 has no daily-return predictors, market residuals, Alpha101 formulas or interactions. Ridge alpha=1000 is fixed for all three models.
