"""Point-in-time *synthetic* equity panel. Not observed market prices/fundamentals."""
from pathlib import Path
import numpy as np
import pandas as pd

SECTORS = ("Technology", "Financials", "Healthcare", "Industrials", "Consumer", "Energy", "Utilities", "Materials", "Communications", "Real Estate")
REQUIRED = {"date", "ticker", "sector", "close", "volume"}


def generate_synthetic_data(path: str | Path, n_assets: int = 80, n_days: int = 740, seed: int = 42) -> pd.DataFrame:
    if n_assets < 30 or n_days < 260:
        raise ValueError("Use at least 30 assets and 260 business days for a meaningful demo")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    market_vol = np.where((np.arange(n_days)//110) % 2 == 0, 0.009, 0.015)
    market = rng.normal(0.00025, market_vol)
    sector_returns = rng.normal(0, 0.006, (n_days, len(SECTORS))) + market[:, None] * 0.25
    all_stocks = []
    for i in range(n_assets):
        sector = SECTORS[i % len(SECTORS)]
        sector_index = i % len(SECTORS)
        true_beta = rng.uniform(0.55, 1.5)
        noise = rng.normal(0, rng.uniform(0.008, 0.020), n_days)
        residual = np.zeros(n_days)
        # Several different plausible (simulated) time-series structures; no alpha guarantee.
        phi = rng.uniform(-0.15, 0.15)
        for t in range(1, n_days):
            residual[t] = phi * residual[t-1] + noise[t]
        ret = np.clip(0.0001 + true_beta * market + sector_returns[:, sector_index] + residual, -0.24, 0.24)
        close = rng.uniform(15, 200) * np.cumprod(1 + ret)
        base_volume = rng.uniform(450_000, 5_000_000)
        volume = np.maximum(1, np.round(base_volume * np.exp(rng.normal(0, 0.35, n_days)))).astype('int64')

        # Quarterly simulated, observable-as-of-t fundamentals; NOT genuine disclosures.
        quarters = np.arange(n_days) // 63
        count_q = int(quarters.max()) + 1
        ey = np.clip(rng.uniform(0.025, 0.11) + np.cumsum(rng.normal(0, 0.003, count_q)), -0.025, 0.20)
        bp = np.clip(rng.uniform(0.12, 0.9) + np.cumsum(rng.normal(0, 0.025, count_q)), 0.02, 1.6)
        sp = np.clip(rng.uniform(0.2, 1.8) + np.cumsum(rng.normal(0, 0.05, count_q)), 0.05, 3.0)
        all_stocks.append(pd.DataFrame({
            "date": dates, "ticker": f"SYN{i+1:03d}", "sector": sector,
            "close": close, "volume": volume,
            "earnings_yield": ey[quarters], "book_to_price": bp[quarters],
            "sales_to_price": sp[quarters],
        }))
    df = pd.concat(all_stocks, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.8f")
    return df


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker records")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing required data")
    if (df["close"] <= 0).any() or (df["volume"] <= 0).any():
        raise ValueError("Price and volume must be positive")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
