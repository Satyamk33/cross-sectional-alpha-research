"""Leakage-aware historical features; optional 3 simulated fundamental factors."""
import numpy as np
import pandas as pd


def engineer_factors(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = prices.sort_values(["ticker", "date"]).copy().reset_index(drop=True)
    ticker = df["ticker"]
    grp = df.groupby("ticker", sort=False)
    df["ret_1d"] = grp["close"].pct_change(fill_method=None)
    df["dollar_volume"] = df["close"] * df["volume"]
    market = df.groupby("date")["ret_1d"].mean()  # equal-weighted *synthetic* market proxy
    df["market_ret"] = df["date"].map(market)
    df["sector_ret"] = df.groupby(["date", "sector"])["ret_1d"].transform("mean")
    features = []

    def add(name, series):
        df[name] = pd.Series(series, index=df.index).replace([np.inf, -np.inf], np.nan)
        features.append(name)

    def rolling(col, w, what="mean"):
        g = df.groupby("ticker", sort=False)[col]
        return g.transform(lambda x: getattr(x.rolling(w, min_periods=w), what)())

    def pct(w):
        return df.groupby("ticker", sort=False)["close"].pct_change(w, fill_method=None)

    for w in (5, 10, 21, 63, 126):
        add(f"momentum_{w}d", pct(w))
    for w, skip in ((21, 5), (63, 5), (126, 21)):
        add(f"momentum_skip_{w}_{skip}d", grp["close"].shift(skip) / grp["close"].shift(w) - 1)
    for w in (1, 2, 3, 5, 10):
        add(f"reversal_{w}d", -pct(w))
    for w in (5, 10, 21, 63):
        add(f"volatility_{w}d", rolling("ret_1d", w, "std"))
    df["negative_return"] = df["ret_1d"].clip(upper=0)
    for w in (10, 21, 63):
        add(f"downside_volatility_{w}d", rolling("negative_return", w, "std"))
    for w in (21, 63):
        add(f"return_skew_{w}d", rolling("ret_1d", w, "skew"))
    df["adv_21d"] = rolling("dollar_volume", 21)
    avg_volume_63 = rolling("volume", 63)
    for w in (5, 10, 21):
        add(f"relative_volume_{w}d", rolling("volume", w) / avg_volume_63.replace(0, np.nan))
    for short, long in ((1, 21), (5, 21), (10, 63)):
        volshort = df["volume"] if short == 1 else rolling("volume", short)
        add(f"volume_surprise_{short}_{long}d", volshort / rolling("volume", long).replace(0, np.nan) - 1)
    for w in (5, 21, 63):
        add(f"dollar_volume_{w}d", np.log1p(rolling("dollar_volume", w)))
    df["amihud"] = df["ret_1d"].abs() / df["dollar_volume"].replace(0, np.nan)
    for w in (5, 21, 63):
        add(f"amihud_{w}d", np.log1p(rolling("amihud", w) * 1e9))
    for w in (10, 21, 63):
        add(f"distance_sma_{w}d", df["close"] / rolling("close", w).replace(0, np.nan) - 1)
    for w in (10, 21, 63):
        add(f"drawdown_{w}d", df["close"] / rolling("close", w, "max") - 1)

    # Rolling beta, estimated using only returns available by decision-date close.
    window = 63
    roll_m = rolling("market_ret", window)
    roll_r = rolling("ret_1d", window)
    cov = rolling_helper(df["ret_1d"] * df["market_ret"], ticker, window) - roll_r * roll_m
    var = rolling_helper(df["market_ret"] ** 2, ticker, window) - roll_m ** 2
    df["beta_63d"] = cov / var.where(var > 1e-12)
    add("estimated_beta_63d", df["beta_63d"])
    for w in (21, 63):
        asset_cum = df.groupby("ticker")["ret_1d"].transform(lambda s: (1+s).rolling(w, min_periods=w).apply(np.prod, raw=True) - 1)
        mkt_cum = df.groupby("ticker")["market_ret"].transform(lambda s: (1+s).rolling(w, min_periods=w).apply(np.prod, raw=True) - 1)
        sector_cum = df.groupby("ticker")["sector_ret"].transform(lambda s: (1+s).rolling(w, min_periods=w).apply(np.prod, raw=True) - 1)
        add(f"residual_momentum_{w}d", asset_cum - df["beta_63d"] * mkt_cum)
        add(f"sector_relative_momentum_{w}d", asset_cum - sector_cum)
    # Historical rolling lag-one autocorrelation, one more way to study reversal.
    df["lagged_ret"] = df.groupby("ticker")["ret_1d"].shift(1)
    for w in (21, 63):
        a = rolling("ret_1d", w)
        b = rolling("lagged_ret", w)
        cov_ab = rolling_helper(df["ret_1d"] * df["lagged_ret"], ticker, w) - a*b
        va = rolling_helper(df["ret_1d"]**2, ticker, w) - a*a
        vb = rolling_helper(df["lagged_ret"]**2, ticker, w) - b*b
        add(f"return_autocorr_{w}d", cov_ab / np.sqrt(va.clip(lower=1e-12) * vb.clip(lower=1e-12)))
    for col in ("earnings_yield", "book_to_price", "sales_to_price"):
        if col in df.columns:
            add(f"fundamental_{col}", df[col])

    # Labels: subsequent close-to-close returns, never used for score engineering.
    for h in (1, 5, 10, 21):
        df[f"fwd_return_{h}d"] = df.groupby("ticker")["close"].shift(-h) / df["close"] - 1
    return df, features


def rolling_helper(values: pd.Series, group: pd.Series, window: int) -> pd.Series:
    return values.groupby(group, sort=False).transform(lambda s: s.rolling(window, min_periods=window).mean())
