"""Build verified tables and combined figures for report.tex from completed runs.

Run from the project root: python utils/build_report_assets.py
No models are retrained and no experiment outputs are modified.
"""
from pathlib import Path
import sys
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils.experiment import market_inputs
from utils.backtest import run_ledger, performance

RUNS = {
    "M1": "model1_20261107_223125_445454",
    "M2": "model2_20261108_121514_275909",
    "M3": "model3_20261108_134524_579273",
    "M4": "model4_20261107_225220_083066",
    "Ablation": "model4_ablation_20261108_121514_275909",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", action="store_true", help="Use the latest completed run for each current configuration")
    args = parser.parse_args()
    if args.latest:
        for key in RUNS:
            model = "model4_ablation" if key == "Ablation" else f"model{key[1:]}"
            candidates = []
            for meta in (ROOT / "output").glob("*/run_info.json"):
                info = json.loads(meta.read_text(encoding="utf-8"))
                if info.get("model") == model and info.get("status") == "completed":
                    candidates.append(meta.parent.name)
            if not candidates:
                raise FileNotFoundError(f"No completed run found for {model}")
            RUNS[key] = max(candidates)
    out = ROOT / "report_assets"
    out.mkdir(exist_ok=True)
    frames, infos = {}, {}
    for key, folder in RUNS.items():
        path = ROOT / "output" / folder
        frames[key] = pd.read_csv(path / "metrics.csv")
        infos[key] = json.loads((path / "run_info.json").read_text())
        assert infos[key]["status"] == "completed"
    def row(key, period="test_2024_2025", cost=5, model=None, year=None):
        frame = frames[key]
        model = model or f"model{key[1:]}"
        mask = frame.period.eq(period) & frame.cost_bps.eq(cost) & frame.model.eq(model)
        mask &= frame.year.isna() if year is None else frame.year.eq(year)
        return frame[mask].iloc[0]
    all_main = ["M1", "M2", "M3", "M4", "SPY"]
    def mainrow(key, **kwargs):
        return row("M4", model="SPY", **kwargs) if key == "SPY" else row(key, **kwargs)
    def pct(value): return f"{100 * value:.2f}"
    def number(value): return f"{value:.3f}"
    def esc(value):
        return str(value).replace("_", r"\_").replace("%", r"\%")
    tables = []
    def table(name, headers, rows, alignment=None):
        alignment = alignment or "l" + "r" * (len(headers) - 1)
        content = [rf"\newcommand{{\{name}}}{{%", rf"\begin{{tabular}}{{{alignment}}}\toprule",
                   " & ".join(headers) + r"\\\midrule"]
        content.extend(" & ".join(map(str, values)) + r"\\" for values in rows)
        content.append(r"\bottomrule\end{tabular}}")
        tables.append("\n".join(content))

    table("MainTestTable", ["Model", "Total (\\%)", "CAGR (\\%)", "Arith. (\\%)", "Vol. (\\%)", "Sharpe", "Max DD (\\%)"],
          [[key, *[pct(mainrow(key)[field]) for field in ["total_return", "cagr", "annualized_arithmetic_return", "annualized_volatility"]],
            number(mainrow(key).sharpe_rf0), pct(mainrow(key).max_drawdown)] for key in all_main])
    periods = [f"validation_{year}" for year in [2021, 2022, 2023]]
    table("ValidationTable", ["Model", "CAGR 2021", "CAGR 2022", "CAGR 2023", "SR 2021", "SR 2022", "SR 2023"],
          [[key, *[pct(mainrow(key, period=p).cagr) for p in periods],
            *[number(mainrow(key, period=p).sharpe_rf0) for p in periods]] for key in all_main])
    table("AnnualTable", ["Model", "2024 return (\\%)", "2024 SR", "2025 return (\\%)", "2025 SR"],
          [[key, pct(mainrow(key, year=2024).total_return), number(mainrow(key, year=2024).sharpe_rf0),
            pct(mainrow(key, year=2025).total_return), number(mainrow(key, year=2025).sharpe_rf0)] for key in all_main])
    table("PredictionTable", ["Model", "IC", "Rank IC", "ICIR", "$R^2$", "Coverage (\\%)"],
          [[key, f"{row(key).mean_ic:.5f}", f"{row(key).mean_rank_ic:.5f}", number(row(key).icir),
            f"{row(key).standardized_target_r2_vs_zero:.6f}", f"{100 * row(key).label_coverage:.4f}"] for key in all_main[:-1]])
    table("CostTable", ["Model", "0 bps CAGR", "5 bps CAGR", "10 bps CAGR", "0--5 drag (pp)"],
          [[key, *[pct(mainrow(key, cost=c).cagr) for c in [0, 5, 10]],
            pct(mainrow(key, cost=0).cagr - mainrow(key).cagr)] for key in all_main])
    table("ExecutionTable", ["Model", "Turnover (\\%)", "Fees (USD)", "Missing buys", "Terminal events", "Max trade/ADV (\\%)"],
          [[key, pct(mainrow(key).mean_daily_one_sided_turnover), f"{mainrow(key).trading_cost_dollars:,.0f}",
            str(int(mainrow(key).missing_entry_orders)), str(int(mainrow(key).terminal_position_events)),
            f"{100 * mainrow(key).max_trade_adv_ratio:.3f}"] for key in all_main])
    search = infos["Ablation"]["validation_search"]
    table("SearchTable", ["Family", "$\\alpha$", "$R^2_{2021}$", "$R^2_{2022}$", "$R^2_{2023}$", "Mean $R^2$"],
          [[item["estimator"], f"{item['alpha']:g}", *[f"{f['r2']:.6f}" for f in item["folds"]],
            f"{item['mean_validation_r2']:.6f}"] for item in search])
    ablation_rows = [["Ridge (fixed)", "1000", "48", pct(row("M4").cagr), pct(row("M4").annualized_volatility),
                      number(row("M4").sharpe_rf0), pct(row("M4").max_drawdown)]]
    for name in ["Ridge", "Lasso", "ElasticNet"]:
        r = row("Ablation", model=name)
        alpha = infos["Ablation"]["selected_parameters"][name]["alpha"]
        ablation_rows.append([name + " (selected)", f"{alpha:g}", str(int(r.active_features)), pct(r.cagr),
                              pct(r.annualized_volatility), number(r.sharpe_rf0), pct(r.max_drawdown)])
    table("AblationTable", ["Estimator", "$\\alpha$", "$k$", "CAGR (\\%)", "Vol. (\\%)", "Sharpe", "Max DD (\\%)"], ablation_rows)
    table("AblationPredictionTable", ["Estimator", "IC", "Rank IC", "$R^2$", "Turnover (\\%)"],
          [[name, f"{row('Ablation', model=name).mean_ic:.5f}", f"{row('Ablation', model=name).mean_rank_ic:.5f}",
            f"{row('Ablation', model=name).standardized_target_r2_vs_zero:.6f}",
            pct(row('Ablation', model=name).mean_daily_one_sided_turnover)] for name in ["Ridge", "Lasso", "ElasticNet"]])
    bundles = joblib.load(ROOT / "output" / RUNS["Ablation"] / "model4_ablation.joblib")
    table("SparseTable", ["Estimator", "Retained predictor", "Coefficient"],
          [[name, esc(feature), f"{coefficient:.6f}"] for name in ["Lasso", "ElasticNet"]
           for feature, coefficient in zip(bundles[name]["feature_names"], bundles[name]["model"].coef_)
           if abs(coefficient) > 1e-12], "llr")
    table("TrainingTable", ["Evaluation", "Cutoff", "Rows", "Signal dates", "Last training signal"],
          [[esc(fold["period"]), fold["label_exit_cutoff"], f"{fold['rows']:,}", str(fold["dates"]), fold["last_signal"]]
           for fold in infos["M4"]["training_folds"]])
    table("VersionTable", ["Package", "Version"], [[esc(k), v] for k, v in infos["M4"]["versions"].items()], "ll")
    table("RunTable", ["Configuration", "Frozen result directory"],
          [[key, r"\path{output/" + folder + "}"] for key, folder in RUNS.items()], r"lp{0.77\linewidth}")
    (out / "tables.tex").write_text("\n\n".join(tables) + "\n", encoding="utf-8")

    plt.rcParams.update({"figure.dpi": 180, "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2})
    colors = {"M1": "#7b3294", "M2": "#008b8b", "M3": "#d88916", "M4": "#1f5aa6", "SPY": "#262626",
              "Ridge": "#1f5aa6", "Lasso": "#bf4b45", "ElasticNet": "#39916b", "Fixed Ridge": "#808080"}
    source = pd.read_parquet(ROOT / "sp500/panel/daily.parquet", columns=["date", "security_id", "ticker", "is_benchmark",
        "adj_open", "adj_close", "delist_ret", "vwap", "volume"], filters=[("date", "<=", pd.Timestamp("2025-12-31"))])
    books = {}
    for key, folder in RUNS.items():
        pred = pd.read_parquet(ROOT / "output" / folder / "predictions.parquet")
        pred = pred[pred.period.eq("test_2024_2025")]
        dates = pd.DatetimeIndex(sorted(pred.date.unique()))
        assets = sorted(set(pred.security_id) | {"SPY"})
        market = market_inputs(source, dates, assets)
        base = pred.drop_duplicates(["date", "security_id"])
        market["universe"] = base.assign(eligible=True).pivot(index="date", columns="security_id", values="eligible").reindex(index=dates, columns=assets).eq(True).to_numpy()
        groups = pred.groupby("model", sort=False) if key == "Ablation" else [(key, pred)]
        for name, frame in groups:
            scores = frame.pivot(index="date", columns="security_id", values="score").reindex(index=dates, columns=assets).to_numpy()
            book = run_ledger(market, scores, "top50_drop5", 5, 0.)[0]
            expected = row(key, model=name) if key == "Ablation" else row(key)
            actual = performance(book)
            for field in ["total_return", "cagr", "max_drawdown", "sharpe_rf0"]:
                np.testing.assert_allclose(actual[field], expected[field], rtol=1e-10, atol=1e-12)
            books[name] = book
        if key == "M4":
            books["SPY"] = run_ledger(market, np.zeros_like(scores), "spy_buy_hold", 5, 0.)[0]

    def wealth_plot(names, filename):
        fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
        for name in names:
            book = books["M4"] if name == "Fixed Ridge" else books[name]
            dates = pd.DatetimeIndex([book.entry_date.iloc[0], *book.exit_date])
            wealth = np.r_[1., book.end_nav.to_numpy() / 1e6]
            style = "--" if name == "Fixed Ridge" else "-"
            axes[0].plot(dates, wealth, label=name, color=colors[name], ls=style, lw=1.6)
            axes[1].plot(dates, wealth / np.maximum.accumulate(wealth) - 1, color=colors[name], ls=style, lw=1.3)
        axes[0].set(ylabel="Net wealth (initial = 1)")
        axes[1].set(ylabel="Drawdown", xlabel="Valuation date")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(ncol=3, fontsize=11.5, loc="upper left")
        fig.tight_layout(); fig.savefig(out / filename, bbox_inches="tight"); plt.close(fig)
    wealth_plot(all_main, "main_performance.png")
    wealth_plot(["Ridge", "Lasso", "ElasticNet", "Fixed Ridge", "SPY"], "ablation_performance.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    x = np.arange(3); width = .15
    for i, key in enumerate(all_main):
        for ax, field in zip(axes, ["cagr", "sharpe_rf0"]):
            ax.bar(x + (i - 2) * width, [mainrow(key, period=p)[field] for p in periods], width, color=colors[key], label=key)
            ax.set_xticks(x, ["2021", "2022", "2023"])
    axes[0].set(title="Validation CAGR", ylabel="Annualized return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(title="Validation Sharpe", ylabel="Sharpe (rf = 0)")
    axes[1].legend(ncol=3, fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "validation.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for key in all_main[:-1]:
        axes[0].plot([0, 5, 10], [mainrow(key, cost=c).cagr for c in [0, 5, 10]], marker="o", color=colors[key], label=key)
    axes[0].plot([0, 5, 10], [mainrow("SPY", cost=c).cagr for c in [0, 5, 10]], marker="o", color=colors["SPY"], label="SPY")
    axes[0].set(title="Main models", xlabel="Cost per side (bps)", ylabel="CAGR")
    for key in ["Ridge", "Lasso", "ElasticNet"]:
        axes[1].plot([0, 5, 10], [row("Ablation", model=key, cost=c).cagr for c in [0, 5, 10]], marker="o", color=colors[key], label=key)
    axes[1].set(title="48-factor regularization ablation", xlabel="Cost per side (bps)", ylabel="CAGR")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "costs.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    all_periods = [*periods, "test_2024_2025"]
    for i, key in enumerate(all_main[:-1]):
        for ax, field, scale in [(axes[0], "standardized_target_r2_vs_zero", 1e4), (axes[1], "mean_rank_ic", 1)]:
            ax.bar(np.arange(4) + (i - 1.5) * .18, [row(key, period=p)[field] * scale for p in all_periods], .18, color=colors[key], label=key)
            ax.set_xticks(np.arange(4), ["2021", "2022", "2023", "2024-25"])
            ax.axhline(0, color="black", lw=.7)
    axes[0].set(title="Prediction fit", ylabel="$10^4 \\times R^2$ (not return bps)")
    axes[1].set(title="Cross-sectional ranking", ylabel="Mean daily Rank IC")
    axes[1].legend(ncol=2, fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "prediction.png", bbox_inches="tight"); plt.close(fig)

    fixed = joblib.load(ROOT / "output" / RUNS["M4"] / "model4.joblib")
    names = fixed["feature_names"]
    chosen = set(np.argsort(np.abs(fixed["model"].coef_))[-12:])
    for key in ["Lasso", "ElasticNet"]:
        chosen.update(np.flatnonzero(np.abs(bundles[key]["model"].coef_) > 1e-12))
    indices = sorted(chosen, key=lambda j: abs(fixed["model"].coef_[j]), reverse=True)
    matrix = np.array([fixed["model"].coef_, bundles["Ridge"]["model"].coef_, bundles["Lasso"]["model"].coef_, bundles["ElasticNet"]["model"].coef_])[:, indices].T
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    limit = np.max(np.abs(matrix))
    heat = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(np.arange(len(indices)), [names[j] for j in indices], fontsize=11.5)
    ax.set_xticks(np.arange(4), ["Fixed Ridge", "Selected Ridge", "Lasso", "Elastic Net"], fontsize=11.5)
    ax.grid(False)
    fig.colorbar(heat, ax=ax, shrink=.8, label="Coefficient on standardized input")
    fig.tight_layout(); fig.savefig(out / "coefficients.png", bbox_inches="tight"); plt.close(fig)
    print("Verified report assets generated in report_assets/; backtests reproduce stored metrics.")


if __name__ == "__main__":
    main()
