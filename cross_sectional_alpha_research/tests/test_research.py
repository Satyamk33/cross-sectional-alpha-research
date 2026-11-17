"""No-leakage smoke tests and neutrality/cost accounting tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import generate_synthetic_data, load_prices
from src.factors import engineer_factors
from src.research import walk_forward_scores, decile_analysis
from src.portfolio import neutral_target, signal_neutrality_diagnostics, backtest


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.TemporaryDirectory()
        file = Path(cls.workdir.name)/"test.csv"
        generate_synthetic_data(file, n_assets=40, n_days=310, seed=9)
        cls.raw = load_prices(file)
        cls.panel, cls.factors = engineer_factors(cls.raw)

    @classmethod
    def tearDownClass(cls):
        cls.workdir.cleanup()

    def test_feature_count_and_finite_demo_labels(self):
        self.assertEqual(len(self.factors), 50)
        self.assertEqual(len(set(self.factors)), 50)
        self.assertTrue(self.panel["fwd_return_5d"].notna().sum() > 100)

    def test_changing_future_prices_does_not_change_past_features(self):
        cutoff = self.raw["date"].unique()[175]
        earlier = self.panel.loc[self.panel["date"] < cutoff, ["ticker", "date"]+self.factors].reset_index(drop=True)
        modified = self.raw.copy()
        modified.loc[modified["date"] >= cutoff, "close"] *= 1.25
        changed, _ = engineer_factors(modified)
        earlier_changed = changed.loc[changed["date"] < cutoff, ["ticker", "date"]+self.factors].reset_index(drop=True)
        pd.testing.assert_frame_equal(earlier, earlier_changed)

    def test_weights_are_sector_and_beta_neutral(self):
        sample = self.panel.loc[self.panel["date"] == sorted(self.panel["date"].unique())[-30]].copy()
        sample["alpha_score"] = np.linspace(-1, 1, len(sample))
        weights = neutral_target(sample)
        diag = signal_neutrality_diagnostics(sample, weights)
        self.assertAlmostEqual(diag["net_exposure"], 0, places=9)
        self.assertAlmostEqual(diag["gross_exposure"], 1, places=9)
        self.assertLess(abs(diag["estimated_market_beta_exposure"]), 1e-9)
        self.assertLess(diag["max_absolute_sector_net_exposure"], 1e-9)

    def test_walk_forward_and_costs(self):
        scored, log, _, _ = walk_forward_scores(self.panel, self.factors, lookback_days=180, top_n=6)
        self.assertFalse(log.empty)
        self.assertTrue(log["training_last_signal_date"].lt(log["refit_date"]).all())
        deciles, _ = decile_analysis(scored)
        self.assertEqual(len(deciles), 10)
        curves, metrics, positions, _, _ = backtest(scored)
        self.assertTrue(np.isfinite(curves["equity"]).all())
        self.assertTrue((curves[["trading_cost", "borrow_cost"]] >= 0).all().all())
        self.assertLess(metrics["maximum_signal_absolute_sector_exposure"], 1e-8)
        self.assertLess(metrics["maximum_signal_absolute_beta_exposure"], 1e-8)
        self.assertLess(metrics["maximum_signal_absolute_net_exposure"], 1e-8)
        self.assertEqual(len(positions), len(curves))


if __name__ == "__main__":
    unittest.main()
