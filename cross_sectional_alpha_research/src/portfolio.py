"""Sector/beta-neutral portfolio weights and illustrative transaction-cost backtest."""
import numpy as np
import pandas as pd


def neutral_target(day: pd.DataFrame, tail_fraction: float = 0.10, gross: float = 1.0) -> pd.Series:
    """Project initial decile long/short weights onto the sector and beta nullspace.

    Constraints apply at rebalance decision time, using rolling beta estimates.
    With sector neutrality, dollar neutrality is implied because sectors partition assets.
    Projection introduces smaller non-tail holdings. No explicit leverage/borrow constraint.
    """
    indexed = day.set_index("ticker").sort_index()
    valid = indexed.dropna(subset=["alpha_score", "beta_63d", "sector"])
    out = pd.Series(0., index=indexed.index, dtype=float, name="weight")
    if len(valid) < 30 or valid["sector"].nunique() < 2:
        return out
    k = max(3, int(len(valid)*tail_fraction))
    ranked = valid["alpha_score"].sort_values(kind="mergesort")
    initial = pd.Series(0., index=valid.index)
    initial.loc[ranked.index[:k]] = -0.5 / k
    initial.loc[ranked.index[-k:]] = 0.5 / k
    sector_dummies = pd.get_dummies(valid["sector"], dtype=float).to_numpy()
    beta = valid["beta_63d"].to_numpy(dtype=float).reshape(-1, 1)
    # Normalizing beta improves numerical conditioning without changing the constraint.
    beta_norm = beta / max(float(beta.std()), 0.01)
    constraints = np.column_stack([sector_dummies, beta_norm])
    unscaled = initial.to_numpy() - constraints @ np.linalg.lstsq(
        constraints, initial.to_numpy(), rcond=None)[0]
    total_gross = float(np.abs(unscaled).sum())
    if total_gross < 1e-10:
        return out
    out.loc[valid.index] = unscaled * gross / total_gross
    return out


def signal_neutrality_diagnostics(day: pd.DataFrame, weights: pd.Series) -> dict:
    indexed = day.set_index("ticker").reindex(weights.index)
    sector_net = weights.groupby(indexed["sector"]).sum()
    return {"net_exposure": float(weights.sum()),
            "gross_exposure": float(weights.abs().sum()),
            "estimated_market_beta_exposure": float((weights*indexed["beta_63d"].fillna(0)).sum()),
            "max_absolute_sector_net_exposure": float(sector_net.abs().max())}


def backtest(oos: pd.DataFrame, rebalance_every: int = 5, execution_lag: int = 2,
             commission_bps: float = 1., half_spread_bps: float = 3.,
             impact_coefficient_bps: float = 25., short_borrow_annual: float = 0.02,
             assumed_capital: float = 5_000_000.):
    if rebalance_every <= 0 or execution_lag < 1:
        raise ValueError("Rebalance must be positive; execution lag at least one bar")
    if min(commission_bps, half_spread_bps, impact_coefficient_bps, short_borrow_annual) < 0:
        raise ValueError("Trading cost parameters cannot be negative")
    tickers = sorted(oos["ticker"].unique())
    dates = pd.DatetimeIndex(sorted(oos["date"].unique()), name="date")
    returns = oos.pivot(index="date", columns="ticker", values="ret_1d").reindex(index=dates, columns=tickers).fillna(0.)
    adv = oos.pivot(index="date", columns="ticker", values="adv_21d").reindex(index=dates, columns=tickers).ffill()
    targets = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    diagnostics = []
    for i in range(0, len(dates), rebalance_every):
        day = oos.loc[oos["date"] == dates[i]].copy()
        w = neutral_target(day).reindex(tickers).fillna(0.)
        targets.loc[dates[i]] = w
        diag = signal_neutrality_diagnostics(day, w)
        diag["decision_date"] = str(dates[i].date())
        diagnostics.append(diag)

    # Signal close t -> fill at close t+1 -> earn close-to-close return dated t+2.
    positions = targets.ffill().shift(execution_lag).fillna(0.)
    gross_ret = (positions * returns).sum(axis=1)
    trades = positions.diff().fillna(positions).abs()
    turnover = trades.sum(axis=1)
    participation = (trades * assumed_capital / adv.replace(0., np.nan)).clip(lower=0., upper=1.)
    impact_bps = impact_coefficient_bps * np.sqrt(participation)
    rate = (commission_bps + half_spread_bps + impact_bps) / 1e4
    trading_cost = (trades * rate.fillna((commission_bps+half_spread_bps)/1e4)).sum(axis=1)
    borrow_cost = positions.clip(upper=0).abs().sum(axis=1) * (short_borrow_annual / 252.)
    net_ret = gross_ret - trading_cost - borrow_cost
    equity = (1 + net_ret).cumprod()
    drawdown = equity / equity.cummax() - 1
    curves = pd.DataFrame({"gross_return": gross_ret, "turnover": turnover,
        "trading_cost": trading_cost, "borrow_cost": borrow_cost,
        "net_return": net_ret, "equity": equity, "drawdown": drawdown})
    std = float(net_ret.std(ddof=1))
    metrics = {"trading_days": int(len(dates)),
        "cumulative_gross_return_compounded": float((1+gross_ret).prod()-1),
        "cumulative_net_return": float(equity.iloc[-1]-1),
        "annualized_sharpe": float(np.sqrt(252)*net_ret.mean()/std) if std > 0 else None,
        "annualized_volatility": float(std*np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "average_daily_turnover": float(turnover.mean()),
        "total_trading_cost_return_units": float(trading_cost.sum()),
        "total_borrow_cost_return_units": float(borrow_cost.sum()),
        "rebalance_every_sessions": rebalance_every,
        "execution_lag_sessions": execution_lag,
        "commission_bps_per_side": commission_bps,
        "half_spread_bps_per_side": half_spread_bps,
        "impact_coefficient_bps": impact_coefficient_bps,
        "short_borrow_annual_rate": short_borrow_annual,
        "assumed_capital_usd": assumed_capital,
        "maximum_signal_absolute_net_exposure": float(max(abs(d["net_exposure"]) for d in diagnostics)),
        "maximum_signal_absolute_sector_exposure": float(max(d["max_absolute_sector_net_exposure"] for d in diagnostics)),
        "maximum_signal_absolute_beta_exposure": float(max(abs(d["estimated_market_beta_exposure"]) for d in diagnostics)),
    }
    return curves, metrics, positions, targets.dropna(how="all"), pd.DataFrame(diagnostics)
