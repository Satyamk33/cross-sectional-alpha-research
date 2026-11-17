"""Reproducible synthetic-data cross-sectional equity alpha research demonstration."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.data import generate_synthetic_data, load_prices
from src.factors import engineer_factors
from src.research import walk_forward_scores, daily_rank_ic, decile_analysis
from src.portfolio import backtest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_equities.csv")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--generate", action="store_true", help="Regenerate 100% synthetic prices")
    parser.add_argument("--n-assets", type=int, default=80)
    parser.add_argument("--n-days", type=int, default=740)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebalance", type=int, default=5)
    args = parser.parse_args(argv)
    data = Path(args.data)
    if args.generate or not data.exists():
        generate_synthetic_data(data, n_assets=args.n_assets, n_days=args.n_days, seed=args.seed)
    report = Path(args.reports)
    report.mkdir(parents=True, exist_ok=True)
    raw = load_prices(data)
    synthetic = raw["ticker"].str.startswith("SYN").all()
    panel, factors = engineer_factors(raw)
    scored, log, ic_all, test_start = walk_forward_scores(panel, factors)
    if log.empty:
        raise RuntimeError("Could not select any predictive factors from training history")
    q, q_daily = decile_analysis(scored, n=10)
    curves, metrics, positions, targets, diagnostics = backtest(scored, rebalance_every=args.rebalance)
    test_ic = daily_rank_ic(scored, factors, horizon=5)
    ic_summary = pd.DataFrame({"factor": factors,
        "oos_rank_ic_mean": test_ic.mean().to_numpy(),
        "oos_rank_ic_std": test_ic.std().to_numpy(),
        "oos_ic_dates": test_ic.count().to_numpy()})
    ic_summary.to_csv(report / "factor_ic_oos.csv", index=False)
    decay_records = []
    for horizon in (1, 5, 10, 21):
        ith = daily_rank_ic(scored, factors, horizon=horizon)
        for factor in factors:
            decay_records.append({"factor": factor, "horizon_sessions": horizon,
                                 "oos_mean_rank_ic": float(ith[factor].mean())})
    pd.DataFrame(decay_records).to_csv(report / "factor_decay_oos.csv", index=False)
    q.to_csv(report / "decile_summary.csv", index=False)
    q_daily.to_csv(report / "daily_decile_returns.csv", index_label="date")
    curves.to_csv(report / "equity_curve.csv", index_label="date")
    positions.to_csv(report / "positions.csv", index_label="date")
    targets.to_csv(report / "rebalance_targets.csv", index_label="date")
    diagnostics.to_csv(report / "neutrality_checks.csv", index=False)
    log.to_csv(report / "walk_forward_selection.csv", index=False)
    metrics.update({
        "dataset_type": "SYNTHETIC_DEMONSTRATION" if synthetic else "USER_PROVIDED_UNVERIFIED",
        "n_stocks": int(raw["ticker"].nunique()),
        "n_sectors": int(raw["sector"].nunique()),
        "features_engineered": len(factors),
        "train_validation_design": "Rolling 378-session in-sample IC window, selection refit every 21 sessions; 6-session label embargo",
        "oos_first_date": str(test_start.date()),
        "oos_last_date": str(curves.index.max().date()),
        "mean_oos_5d_decile_spread": float(q_daily["long_short_top_minus_bottom"].mean()),
        "important_warning": "Synthetic securities and fundamentals are fictional. Returns are illustrative and NOT evidence of live alpha or real historical performance.",
    })
    (report / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(curves.index, curves["equity"], label="Net long-short equity")
    ax.axhline(1., linestyle="--", color="gray", linewidth=0.8)
    ax.set(title="SYNTHETIC DEMO — Out-of-sample equity curve" if synthetic else "User-supplied equity curve (unverified)",
           xlabel="Date", ylabel="Equity (start=1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report / "equity_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(q["decile"].astype(str), q["mean_forward_return"])
    ax.set(title="SYNTHETIC DEMO — 5-day forward return by signal decile" if synthetic else "Signal decile analysis",
           xlabel="Decile (1=lowest, 10=highest)", ylabel="Mean 5-day forward return")
    fig.tight_layout()
    fig.savefig(report / "decile_returns.png", dpi=160)
    plt.close(fig)

    strongest = ic_summary.dropna().assign(abs_ic=lambda d: d["oos_rank_ic_mean"].abs()).nlargest(15, "abs_ic")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(strongest["factor"], strongest["oos_rank_ic_mean"])
    ax.invert_yaxis()
    ax.set(title="SYNTHETIC DEMO — Selected diagnostic: largest |OOS rank IC|" if synthetic else "OOS factor rank IC", xlabel="Mean Spearman rank IC")
    fig.tight_layout()
    fig.savefig(report / "factor_ic.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved 13 research outputs to {report.resolve()}")
    return metrics


if __name__ == "__main__":
    main()
