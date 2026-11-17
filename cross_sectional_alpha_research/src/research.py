"""Cross-sectional rank IC, decile tests and walk-forward factor selection."""
import numpy as np
import pandas as pd


def daily_rank_ic(panel: pd.DataFrame, factors: list[str], horizon: int = 5) -> pd.DataFrame:
    """Daily Spearman IC, with per-factor pairwise complete observations."""
    y = f"fwd_return_{horizon}d"
    dates = pd.Index(sorted(panel["date"].unique()), name="date")
    out = {}
    for f in factors:
        sub = panel[["date", f, y]].dropna()
        if sub.empty:
            out[f] = pd.Series(dtype=float)
            continue
        ranks = sub.groupby("date")[[f, y]].rank()
        n = sub.groupby("date")[f].size()
        avgx = ranks[f].groupby(sub["date"]).mean()
        avgy = ranks[y].groupby(sub["date"]).mean()
        x = ranks[f] - sub["date"].map(avgx)
        yy = ranks[y] - sub["date"].map(avgy)
        cov = (x * yy).groupby(sub["date"]).sum()
        xx = (x*x).groupby(sub["date"]).sum()
        ys = (yy*yy).groupby(sub["date"]).sum()
        ic = cov / np.sqrt(xx*ys).replace(0, np.nan)
        ic[n < 10] = np.nan
        out[f] = ic
    return pd.DataFrame(out, index=dates)


def family(name: str) -> str:
    return name.split("_")[0] if not name.startswith("fundamental_") else "fundamental"


def select_factors(ic_training: pd.DataFrame, top_n: int = 8, min_days: int = 65) -> pd.DataFrame:
    stats = pd.DataFrame({"factor": ic_training.columns,
                          "train_ic": ic_training.mean().to_numpy(),
                          "observations": ic_training.count().to_numpy()})
    stats = stats[(stats["observations"] >= min_days) & stats["train_ic"].notna()].copy()
    stats["family"] = stats["factor"].map(family)
    stats["abs_ic"] = stats["train_ic"].abs()
    stats = stats.sort_values(["abs_ic", "factor"], ascending=[False, True])
    # Diversity filter: avoid a selected portfolio containing 8 near-duplicate momentum windows.
    selected = stats.drop_duplicates("family").head(top_n).copy()
    selected["direction"] = np.sign(selected["train_ic"]).astype(int)
    return selected[["factor", "family", "train_ic", "observations", "direction"]].reset_index(drop=True)


def score_cross_section(day: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    day = day.copy()
    if selected.empty:
        day["alpha_score"] = np.nan
        return day
    factors = selected["factor"].tolist()
    ranks = day[factors].rank(method="average", pct=True)
    for factor, direction in zip(factors, selected["direction"]):
        ranks[factor] = (ranks[factor] - 0.5) * direction
    # Require all selected features observed, don't backfill missing data.
    day["alpha_score"] = ranks.mean(axis=1).where(ranks.notna().all(axis=1))
    return day


def walk_forward_scores(panel: pd.DataFrame, factors: list[str],
                        start_frac: float = 0.65, refit_every: int = 21,
                        horizon: int = 5, lookback_days: int = 378,
                        top_n: int = 8):
    dates = pd.Index(sorted(panel["date"].unique()))
    start = max(240, int(len(dates)*start_frac))
    if start + 40 >= len(dates):
        raise ValueError("Data needs more history to evaluate out of sample")
    # Used only on an explicitly older slice when selecting factors.
    ic_all = daily_rank_ic(panel, factors, horizon=horizon)
    selected_log = []
    scored_rows = []
    selected = pd.DataFrame()
    for i in range(start, len(dates)):
        if (i-start) % refit_every == 0:
            # At close date i, 5d target for date i-5 is now known. Leave one more day as embargo.
            cutoff_idx = i - horizon - 1
            left_idx = max(0, cutoff_idx - lookback_days + 1)
            eligible_ic = ic_all.iloc[left_idx:cutoff_idx+1]
            selected = select_factors(eligible_ic, top_n=top_n)
            for _, row in selected.iterrows():
                selected_log.append({"refit_date": str(dates[i].date()),
                    "training_last_signal_date": str(dates[cutoff_idx].date()),
                    "factor": row["factor"], "family": row["family"],
                    "train_ic": float(row["train_ic"]), "direction": int(row["direction"]),
                    "train_observations": int(row["observations"])})
        day = panel[panel["date"] == dates[i]]
        scored_rows.append(score_cross_section(day, selected))
    return pd.concat(scored_rows, ignore_index=True), pd.DataFrame(selected_log), ic_all, dates[start]


def decile_analysis(oos: pd.DataFrame, n: int = 10, horizon: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns are overlapping across dates; diagnostics, NOT independently tradable PnL."""
    y = f"fwd_return_{horizon}d"
    sub = oos.dropna(subset=["alpha_score", y]).copy()
    # Stable tie breaking by ticker; fewer than n valid assets are excluded.
    sub = sub.sort_values(["date", "ticker"])
    size = sub.groupby("date")["ticker"].transform("size")
    sub = sub[size >= n]
    ordered_rank = sub.groupby("date")["alpha_score"].rank(method="first")
    size = sub.groupby("date")["ticker"].transform("size")
    sub["decile"] = np.floor((ordered_rank - 1) * n / size).astype(int) + 1
    by_day = sub.groupby(["date", "decile"])[y].mean().unstack("decile")
    summary = pd.DataFrame({"decile": range(1, n+1),
                            "mean_forward_return": [by_day[d].mean() for d in range(1, n+1)],
                            "observations": [by_day[d].count() for d in range(1, n+1)]})
    by_day["long_short_top_minus_bottom"] = by_day[n] - by_day[1]
    return summary, by_day
