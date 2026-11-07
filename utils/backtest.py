"""Self-financing backtest ledger migrated from the supplied team model.

Signals use finalized day-t data; trades fill at the next adjusted open.
"""
import re

import numpy as np
import pandas as pd

CAPITAL = 1_000_000.


def select_weights(scores, universe, holdings, style, spy_index):
    """Only signal-day scores/eligibility and already held names enter ranks."""
    weights = np.zeros(len(scores))
    if style == "spy_buy_hold":
        weights[spy_index] = 1.
        return weights, (1., 0.)
    pool = np.flatnonzero(universe & np.isfinite(scores))
    pool = pool[pool != spy_index]
    ranked = pool[np.lexsort((pool, -scores[pool]))]
    if not len(ranked):
        return weights, (1., 0.)
    if re.fullmatch(r"long_top[1-9][0-9]*", style) and style not in ["long_top10", "long_top20"]:
        k = min(int(style.removeprefix("long_top")), len(ranked))
        weights[ranked[:k]] = 1 / k
        return weights, (1., 0.)
    if style == "universe_equal_weight":
        weights[ranked] = 1 / len(ranked)
        return weights, (1., 0.)
    drop_rule = re.fullmatch(r"top([1-9][0-9]*)_drop([0-9]+)", style)
    if drop_rule:
        target_count, replacement_count = map(int, drop_rule.groups())
        k = min(target_count, len(ranked))
        held = set(np.flatnonzero(holdings > 1e-12)) & set(ranked)
        # Membership/eligibility exits are mandatory; additions to reach k do
        # not consume the five discretionary replacements.
        kept = sorted(held, key=lambda j: (-scores[j], j))[:k]
        kept_set = set(kept)
        for j in ranked:
            if len(kept) >= k:
                break
            if j not in kept_set:
                kept.append(int(j))
                kept_set.add(int(j))
        for _ in range(replacement_count):
            outsiders = [int(j) for j in ranked if j not in kept_set]
            if not outsiders:
                break
            worst = min(kept, key=lambda j: (scores[j], -j))
            best = outsiders[0]
            if scores[best] <= scores[worst]:
                break
            kept.remove(worst)
            kept.append(best)
            kept_set.remove(worst)
            kept_set.add(best)
        weights[kept] = 1 / k
        return weights, (1., 0.)
    fraction = .1 if style == "long_top10" else .2
    k = max(1, int(np.ceil(fraction * len(ranked))))
    if style == "longshort_20":
        k = min(k, len(ranked) // 2)
        if k:
            weights[ranked[:k]] = .5 / k
            weights[ranked[-k:]] = -.5 / k
        return weights, (.5, .5)
    weights[ranked[:k]] = 1 / k
    if style == "long_top20_spy_hedge":
        weights[spy_index] = -1.
        return weights, (1., 1.)
    if style not in ["long_top20", "long_top10"]:
        raise ValueError(style)
    return weights, (1., 0.)


def execute_rebalance(cash, holdings, marks, opens, target_weights, fee_rate, leg_caps):
    """Solve fees and post-fee allocation together; unfilled weights stay cash.

    Existing untradable holdings are frozen. Their dollar values consume the
    relevant long/short budget; available targets are scaled DOWN if necessary,
    never up to replace an unavailable planned entry with hindsight.
    """
    current = holdings * marks
    equity = float(cash + current.sum())
    if equity < -1e-6:
        raise ValueError("Insolvent portfolio; margin liquidation not modeled")
    tradable = np.isfinite(opens) & (opens > 0)
    frozen = ~tradable
    fixed_long = np.maximum(current[frozen], 0).sum()
    fixed_short = np.maximum(-current[frozen], 0).sum()
    after_fee = max(0., equity)
    desired = current.copy()
    for _ in range(40):
        desired = current.copy()
        planned = target_weights * after_fee
        long_mask, short_mask = tradable & (planned > 0), tradable & (planned < 0)
        desired[tradable] = 0.
        long_budget = max(0., leg_caps[0] * after_fee - fixed_long)
        short_budget = max(0., leg_caps[1] * after_fee - fixed_short)
        long_total = planned[long_mask].sum()
        short_total = -planned[short_mask].sum()
        if long_total > 0:
            desired[long_mask] = planned[long_mask] * min(1., long_budget / long_total)
        if short_total > 0:
            desired[short_mask] = planned[short_mask] * min(1., short_budget / short_total)
        traded = float(np.abs(desired[tradable] - current[tradable]).sum())
        fee = fee_rate * traded
        updated = max(0., equity - fee)
        if abs(updated - after_fee) < 1e-13 * max(1., equity):
            after_fee = updated
            break
        after_fee = updated
    trades = desired - current
    trades[frozen] = 0.
    fee = fee_rate * float(np.abs(trades).sum())
    new_cash = float(cash - trades.sum() - fee)
    new_holdings = holdings.copy()
    new_holdings[tradable] = desired[tradable] / opens[tradable]
    new_holdings[np.abs(new_holdings) < 1e-12] = 0.
    check = new_cash + np.dot(new_holdings, marks)
    np.testing.assert_allclose(check, equity - fee, atol=1e-6, rtol=1e-10)
    if new_cash < -1e-5:
        raise AssertionError("Cash borrowing outside the specified strategy")
    return new_cash, new_holdings, trades, fee


def run_ledger(data, scores, style, cost_bps=5., borrow_rate=.02, terminal_mode="source", save_positions=False, rebalance_every=1):
    opens, closes, terminal = data["open"], data["close"], data["terminal"]
    calendar, assets, spy_index = data["calendar"], data["assets"], data["spy_index"]
    cash = float(data.get("capital", CAPITAL))
    holdings = np.zeros(len(assets))
    marks = data["initial_marks"].copy()
    marks[~np.isfinite(marks)] = 0.
    rows, events, positions = [], [], []
    c = cost_bps / 10000
    for step in range(len(calendar) - 1):
        available = np.isfinite(opens[step]) & (opens[step] > 0)
        marks[available] = opens[step, available]
        start_nav = float(cash + np.dot(holdings, marks))
        weights, caps = select_weights(scores[step], data["universe"][step], holdings, style, spy_index)
        if step % rebalance_every:
            # Fixed scheduled holding: do not generate new orders between
            # rebalances. A held name need not remain in the selection pool.
            weights = holdings * marks / start_nav if start_nav > 0 else np.zeros(len(assets))
        frozen_count = int(np.sum((np.abs(holdings) > 1e-12) & ~available))
        missing_entries = int(np.sum((weights != 0) & ~available & (np.abs(holdings) < 1e-12)))
        cash, holdings, trades, trade_cost = execute_rebalance(cash, holdings, marks, opens[step], weights, c, caps)
        after_nav = start_nav - trade_cost
        dollar_positions = holdings * marks
        gross = float(np.abs(dollar_positions).sum() / after_nav) if after_nav > 0 else 0.
        net = float(dollar_positions.sum() / after_nav) if after_nav > 0 else 0.
        n_positions = int(np.sum(np.abs(holdings) > 1e-12))
        days = (pd.Timestamp(calendar[step + 1]) - pd.Timestamp(calendar[step])).days
        borrow_cost = borrow_rate * float(np.maximum(-dollar_positions, 0).sum()) * days / 365
        ratios = np.divide(np.abs(trades), data["adv"][step], out=np.zeros(len(assets)),
                           where=np.isfinite(data["adv"][step]) & (data["adv"][step] > 0))
        if save_positions:
            for j in np.flatnonzero(np.abs(holdings) > 1e-12):
                positions.append({"entry_date": calendar[step], "security_id": assets[j],
                                  "weight": dollar_positions[j] / after_nav if after_nav else 0.,
                                  "adjusted_units": holdings[j]})
        close_valid = np.isfinite(closes[step]) & (closes[step] > 0)
        marks[close_valid] = closes[step, close_valid]
        terminal_count = 0
        # Terminal status is used AFTER this session's opening trades, never
        # to avoid buying an eventual failure using future information.
        for j in np.flatnonzero(np.isfinite(terminal[step]) & (np.abs(holdings) > 1e-12)):
            ret = terminal[step, j] if terminal_mode == "source" else 0.
            proceeds = float(holdings[j] * marks[j] * (1 + ret))
            events.append({"date": calendar[step], "security_id": assets[j], "adjusted_units": holdings[j],
                           "last_mark": marks[j], "source_terminal_return": terminal[step, j],
                           "applied_terminal_return": ret, "settlement_cash": proceeds,
                           "opening_dollar_exposure": dollar_positions[j]})
            cash += proceeds
            holdings[j] = 0.
            terminal_count += 1
        cash -= borrow_cost
        next_available = np.isfinite(opens[step + 1]) & (opens[step + 1] > 0)
        marks[next_available] = opens[step + 1, next_available]
        end_nav = float(cash + np.dot(holdings, marks))
        final_cost = 0.
        final_traded = 0.
        if step == len(calendar) - 2:
            final_traded = float(np.abs(holdings[next_available] * opens[step + 1, next_available]).sum())
            final_cost = c * final_traded
            cash += float(np.dot(holdings[next_available], opens[step + 1, next_available])) - final_cost
            holdings[next_available] = 0.
            end_nav = float(cash + np.dot(holdings, marks))
        if end_nav < -1e-6:
            raise ValueError("Negative NAV; unsupported margin insolvency")
        market_end_before_exit_fees = end_nav + borrow_cost + final_cost
        market_return = market_end_before_exit_fees / after_nav - 1 if after_nav > 0 else 0.
        rows.append({"signal_date": data["signal_dates"][step], "entry_date": calendar[step], "exit_date": calendar[step + 1],
            "start_nav": start_nav, "end_nav": end_nav,
            "net_return": end_nav / start_nav - 1 if start_nav > 0 else 0., "market_return": market_return,
            "trade_cost": trade_cost + final_cost, "borrow_cost": borrow_cost,
            "traded_notional": float(np.abs(trades).sum()) + final_traded,
            "traded_fraction": (float(np.abs(trades).sum()) + final_traded) / start_nav if start_nav > 0 else 0.,
            "gross_exposure": gross, "net_exposure": net, "positions": n_positions,
            "missing_entries": missing_entries, "frozen_holdings": frozen_count,
            "terminal_events": terminal_count, "max_trade_adv_ratio": float(ratios.max()),
            "final_unliquidated_positions": int(np.sum(np.abs(holdings) > 1e-12)) if step == len(calendar) - 2 else 0})
    return pd.DataFrame(rows), pd.DataFrame(events), pd.DataFrame(positions)


def performance(daily, initial=CAPITAL):
    r = daily.net_return.to_numpy()
    wealth = np.r_[initial, daily.end_nav.to_numpy()]
    final = wealth[-1] / initial
    volatility = np.std(r, ddof=1) * np.sqrt(252) if len(r) > 1 else np.nan
    mean = np.mean(r) * 252
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {"days": len(r), "total_return": final - 1, "cagr": final ** (252 / len(r)) - 1,
            "annualized_arithmetic_return": mean, "annualized_volatility": volatility,
            "sharpe_rf0": mean / volatility if volatility > 0 else np.nan,
            "max_drawdown": float(drawdown.min()), "mean_daily_traded_fraction": daily.traded_fraction.mean(),
            "mean_daily_one_sided_turnover": daily.traded_fraction.mean() / 2,
            "trading_cost_dollars": daily.trade_cost.sum(), "borrow_cost_dollars": daily.borrow_cost.sum(),
            "mean_gross_exposure": daily.gross_exposure.mean(), "mean_net_exposure": daily.net_exposure.mean(),
            "missing_entry_orders": int(daily.missing_entries.sum()), "frozen_holding_days": int(daily.frozen_holdings.sum()),
            "terminal_position_events": int(daily.terminal_events.sum()),
            "max_trade_adv_ratio": daily.max_trade_adv_ratio.max(),
            "final_unliquidated_positions": int(daily.final_unliquidated_positions.iloc[-1])}
