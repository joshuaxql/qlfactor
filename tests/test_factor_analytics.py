"""因子分析链路测试：远期收益（含/不含复权）、IC、分组收益、绩效、去极值/标准化。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qlfactor.factor_engine import Factor


def _make_panel(n_dates: int = 30, n_symbols: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    rows = []
    for sym_i, sym in enumerate(symbols):
        close = 10.0 + np.cumsum(rng.normal(0, 0.2, size=n_dates))
        adj = 1.0 + 0.01 * np.arange(n_dates)
        for d_i, dt in enumerate(dates):
            rows.append(
                {
                    "date": dt,
                    "symbol": sym,
                    "open": close[d_i],
                    "high": close[d_i] + 0.5,
                    "low": close[d_i] - 0.5,
                    "close": close[d_i],
                    "pre_close": close[d_i - 1] if d_i > 0 else close[d_i],
                    "change": 0.0,
                    "pct_chg": 0.0,
                    "volume": 1000 + sym_i * 10,
                    "amount": 10000 + sym_i * 100,
                    "adj_factor": adj[d_i],
                    "turnover_rate": 1.0,
                    "turnover_rate_f": 1.0,
                    "volume_ratio": 1.0,
                    "pe": 10 + sym_i,
                    "pe_ttm": 10 + sym_i,
                    "pb": 1.0,
                    "ps": 2.0,
                    "ps_ttm": 1.8,
                    "dv_ratio": 0.2,
                    "dv_ttm": 0.3,
                    "total_share": 1000 + sym_i,
                    "float_share": 800,
                    "free_share": 700,
                    "total_mv": 10000 + sym_i * 100,
                    "circ_mv": 8000 + sym_i * 80,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


class CrossSectionFactor(Factor):
    """每日截面因子值就是 PE，用于稳定测试。"""

    def calculate(self) -> pd.Series:
        return self.PE.astype(float)


class ClosePriceFactor(Factor):
    """直接使用 CLOSE，便于验证复权是否发生在因子计算前。"""

    def calculate(self) -> pd.Series:
        return self.CLOSE.astype(float)


class ConstantFactor(Factor):
    """返回常数因子，用于触发分组退化场景。"""

    def calculate(self) -> pd.Series:
        return pd.Series(1.0, index=self.data.index)


class TestForwardReturnsAdjust(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _make_panel()
        cls.factor = CrossSectionFactor("xs", cls.data)

    def test_adjust_true_uses_adjusted_close(self):
        result = self.factor.get_clean_factor_and_forward_returns(
            quantiles=5, period=1, adjust=True
        )
        # 复权价 = close * adj_factor，其在每只股票上单调上漂移；
        # 取一个单股一日的样本人工核对一遍。
        sym = "S00"
        adj_close = self.data["close"] * self.data["adj_factor"]
        sym_close = adj_close.xs(sym, level="symbol", drop_level=False).sort_index()
        # next_ret 在 (date_i, sym) 上 = sym_close[date_{i+1}] / sym_close[date_i] - 1
        first_idx = result.xs(sym, level="symbol").index[0]
        first_close = sym_close.loc[(first_idx, sym)]
        next_idx = sym_close.index[sym_close.index.get_loc((first_idx, sym)) + 1]
        expected = sym_close.loc[next_idx] / first_close - 1
        actual = result.loc[(first_idx, sym), "next_ret"]
        self.assertAlmostEqual(expected, actual, places=10)

    def test_adjust_false_uses_raw_close(self):
        result = self.factor.get_clean_factor_and_forward_returns(
            quantiles=5, period=1, adjust=False
        )
        sym = "S00"
        raw_close = (
            self.data["close"].xs(sym, level="symbol", drop_level=False).sort_index()
        )
        first_idx = result.xs(sym, level="symbol").index[0]
        first_close = raw_close.loc[(first_idx, sym)]
        next_idx = raw_close.index[raw_close.index.get_loc((first_idx, sym)) + 1]
        expected = raw_close.loc[next_idx] / first_close - 1
        actual = result.loc[(first_idx, sym), "next_ret"]
        self.assertAlmostEqual(expected, actual, places=10)

    def test_adjust_requires_adj_factor_column(self):
        data = self.data.drop(columns=["adj_factor"])

        class F(Factor):
            def calculate(self) -> pd.Series:
                return self.CLOSE

        f = F("no_adj", data)
        with self.assertRaises(KeyError):
            f.get_clean_factor_and_forward_returns(adjust=True)

    def test_adjust_true_pre_adjusts_factor_input_close(self):
        f = ClosePriceFactor("close_factor", self.data)
        result = f.get_clean_factor_and_forward_returns(
            quantiles=5, period=1, adjust=True
        )
        idx = result.index[0]
        expected = self.data.loc[idx, "close"] * self.data.loc[idx, "adj_factor"]
        actual = result.loc[idx, "factor"]
        self.assertAlmostEqual(expected, actual, places=10)

    def test_adjust_false_keeps_raw_factor_input_close(self):
        f = ClosePriceFactor("close_factor", self.data)
        result = f.get_clean_factor_and_forward_returns(
            quantiles=5, period=1, adjust=False
        )
        idx = result.index[0]
        expected = self.data.loc[idx, "close"]
        actual = result.loc[idx, "factor"]
        self.assertAlmostEqual(expected, actual, places=10)


class TestStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _make_panel()
        cls.factor = CrossSectionFactor("xs", cls.data)
        cls.result = cls.factor.get_clean_factor_and_forward_returns(
            quantiles=5, period=1, adjust=False
        )

    def test_calculate_daily_ic(self):
        daily_ic = self.factor.calculate_daily_ic(self.result)
        # 因子在每日截面上恒定（PE=10..19），收益是随机的；IC 仍可计算
        self.assertEqual(daily_ic.name, "IC")
        # 大多数日期至少有一个非 NaN 值
        self.assertGreater(daily_ic.notna().sum(), 0)

    def test_ic_statistics(self):
        daily_ic = self.factor.calculate_daily_ic(self.result)
        stats = self.factor.ic_statistics_analysis(daily_ic)
        self.assertIn("IC均值", stats.index)
        self.assertIn("IC信息比率(IR,年化)", stats.index)

    def test_calculate_daily_group_ret(self):
        gret = self.factor.calculate_daily_group_ret(self.result)
        self.assertEqual(set(gret.columns), {f"组{i}" for i in range(1, 6)})

    def test_long_short(self):
        gret = self.factor.calculate_daily_group_ret(self.result)
        ls_ret, ls_perf = self.factor.calculate_long_short(gret)
        self.assertEqual(len(ls_ret), len(gret))
        self.assertIn("年化收益", ls_perf.index)

    def test_performance_analysis_zero_vol(self):
        # 用整数避免 0.01 的浮点表示在 std() 后产生极小残差
        ret = pd.Series([1.0] * 10)
        perf = self.factor.performance_analysis(ret)
        self.assertEqual(perf["年化波动率"], 0.0)
        self.assertEqual(perf["夏普比率"], 0)


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _make_panel()
        cls.factor = CrossSectionFactor("xs", cls.data)

    def test_winsorize_3sigma_clips_outliers(self):
        s = pd.Series([1.0, 1.0, 1.0, 1.0, 100.0])
        out = self.factor.winsorize_3sigma(s, n=1)
        self.assertLess(out.iloc[-1], 100.0)
        self.assertEqual(out.iloc[0], 1.0)

    def test_winsorize_mad_clips_outliers(self):
        # 让 MAD>0：样本里有一半偏离中位数
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        out = self.factor.winsorize_mad(s, k=3)
        self.assertLess(out.iloc[-1], 100.0)

    def test_standardize_zscore_zero_std(self):
        s = pd.Series([5.0, 5.0, 5.0])
        out = self.factor.standardize_zscore(s)
        self.assertTrue((out == 0).all())

    def test_standardize_zscore_unit_variance(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = self.factor.standardize_zscore(s)
        self.assertAlmostEqual(out.mean(), 0.0, places=10)
        self.assertAlmostEqual(out.std(), 1.0, places=10)

    def test_calculate_method_is_abstract(self):
        with self.assertRaises(TypeError):
            Factor("x", self.data)  # type: ignore[abstract]


class TestEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _make_panel(n_dates=10, n_symbols=6)

    def test_get_clean_rejects_non_positive_period(self):
        f = CrossSectionFactor("xs", self.data)
        with self.assertRaises(ValueError):
            f.get_clean_factor_and_forward_returns(period=0, adjust=False)
        with self.assertRaises(ValueError):
            f.get_clean_factor_and_forward_returns(period=-1, adjust=False)

    def test_get_clean_raises_when_groups_are_insufficient(self):
        f = ConstantFactor("const", self.data)
        with self.assertRaises(ValueError):
            f.get_clean_factor_and_forward_returns(
                quantiles=5,
                period=1,
                adjust=False,
            )

    def test_calculate_long_short_requires_two_groups(self):
        f = CrossSectionFactor("xs", self.data)
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        group_ret = pd.DataFrame({"组1": [0.01, 0.02, 0.03]}, index=idx)
        with self.assertRaises(ValueError):
            f.calculate_long_short(group_ret, period=1)

    def test_calculate_net_returns_requires_two_groups(self):
        f = CrossSectionFactor("xs", self.data)
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        group_ret = pd.DataFrame({"组1": [0.01, 0.02, 0.03]}, index=idx)
        group_turnover = pd.DataFrame({"组1": [0.5, 0.5, 0.5]}, index=idx)
        with self.assertRaises(ValueError):
            f.calculate_net_returns_with_cost(
                group_ret,
                group_turnover,
                cost_bps=10,
            )

    def test_single_side_cost_is_applied_on_both_trade_sides(self):
        f = CrossSectionFactor("xs", self.data)
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        group_ret = pd.DataFrame(
            {
                "组1": [0.01, 0.02, 0.03],
                "组2": [0.03, 0.04, 0.05],
            },
            index=idx,
        )
        group_turnover = pd.DataFrame(
            {
                "组1": [0.5, 0.5, 0.5],
                "组2": [0.5, 0.5, 0.5],
            },
            index=idx,
        )

        group_net_ret, long_short_gross, long_short_net, _ = (
            f.calculate_net_returns_with_cost(
                group_ret,
                group_turnover,
                cost_bps=10,
            )
        )

        # 10bps 单边成本 -> 双边调仓总成本系数 = 2 * 10 / 10000 = 0.002
        expected_group_cost = group_turnover * 0.002
        pd.testing.assert_series_equal(
            group_net_ret["组1"],
            group_ret["组1"] - expected_group_cost["组1"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            group_net_ret["组2"],
            group_ret["组2"] - expected_group_cost["组2"],
            check_names=False,
        )

        expected_ls_gross = group_ret["组2"] - group_ret["组1"]
        expected_ls_net = expected_ls_gross - (group_turnover["组1"] + group_turnover["组2"]) * 0.002
        pd.testing.assert_series_equal(long_short_gross, expected_ls_gross, check_names=False)
        pd.testing.assert_series_equal(long_short_net, expected_ls_net, check_names=False)

    def test_asymmetric_cost_and_multiplier(self):
        f = CrossSectionFactor("xs", self.data)
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        group_ret = pd.DataFrame(
            {
                "组1": [0.01, 0.02, 0.03],
                "组2": [0.03, 0.04, 0.05],
            },
            index=idx,
        )
        group_turnover = pd.DataFrame(
            {
                "组1": [0.2, 0.3, 0.4],
                "组2": [0.5, 0.6, 0.7],
            },
            index=idx,
        )

        # 买入 8bps + 卖出 12bps，再乘 1.5 倍 => 双边总成本率 0.003
        round_trip_rate = ((8 + 12) / 10000) * 1.5
        group_net_ret, long_short_gross, long_short_net, _ = (
            f.calculate_net_returns_with_cost(
                group_ret,
                group_turnover,
                cost_bps=10,
                buy_cost_bps=8,
                sell_cost_bps=12,
                round_trip_multiplier=1.5,
            )
        )

        expected_group_net = group_ret - group_turnover * round_trip_rate
        expected_ls_gross = group_ret["组2"] - group_ret["组1"]
        expected_ls_net = expected_ls_gross - (
            group_turnover["组1"] + group_turnover["组2"]
        ) * round_trip_rate

        pd.testing.assert_frame_equal(group_net_ret, expected_group_net)
        pd.testing.assert_series_equal(long_short_gross, expected_ls_gross, check_names=False)
        pd.testing.assert_series_equal(long_short_net, expected_ls_net, check_names=False)

    def test_cost_model_rejects_negative_values(self):
        f = CrossSectionFactor("xs", self.data)
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        group_ret = pd.DataFrame(
            {
                "组1": [0.01, 0.02, 0.03],
                "组2": [0.03, 0.04, 0.05],
            },
            index=idx,
        )
        group_turnover = pd.DataFrame(
            {
                "组1": [0.5, 0.5, 0.5],
                "组2": [0.5, 0.5, 0.5],
            },
            index=idx,
        )
        with self.assertRaises(ValueError):
            f.calculate_net_returns_with_cost(
                group_ret,
                group_turnover,
                buy_cost_bps=-1,
            )
        with self.assertRaises(ValueError):
            f.calculate_net_returns_with_cost(
                group_ret,
                group_turnover,
                round_trip_multiplier=-1,
            )

    def test_prepare_factor_cache_refreshes_after_inplace_data_change(self):
        data = self.data.copy()
        f = ClosePriceFactor("close_factor", data)
        before = f._prepare_factor_values(adjust=False)
        idx = before.index[0]
        original = float(before.loc[idx])

        data.loc[idx, "close"] = data.loc[idx, "close"] + 10.0
        after = f._prepare_factor_values(adjust=False)

        self.assertNotEqual(float(after.loc[idx]), original)

    def test_performance_analysis_rejects_non_positive_period(self):
        f = CrossSectionFactor("xs", self.data)
        with self.assertRaises(ValueError):
            f.performance_analysis(pd.Series([0.01, -0.01]), period=0)

    def test_group_turnover_uses_equal_weight_rebalance_formula(self):
        f = CrossSectionFactor("xs", self.data)
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        symbols = ["A", "B", "C", "D"]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        result = pd.DataFrame(index=idx)
        result["factor"] = 1.0
        result["next_ret"] = 0.0
        # day1: 组1={A,B}, 组2={C,D}; day2: 组1={A,B,C,D}
        result["factor_quantile"] = [1, 1, 2, 2, 1, 1, 1, 1]

        group_turnover = f.calculate_group_turnover(result)
        self.assertTrue(np.isnan(group_turnover.loc[pd.Timestamp("2024-01-01"), "组1"]))
        # 组1从2只扩容到4只，重合2只 => 1 - 2 / max(2,4) = 0.5
        self.assertAlmostEqual(group_turnover.loc[pd.Timestamp("2024-01-02"), "组1"], 0.5, places=10)

    def test_group_turnover_respects_missing_middle_dates(self):
        f = CrossSectionFactor("xs", self.data)
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        symbols = ["A", "B"]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        result = pd.DataFrame(index=idx)
        result["factor"] = 1.0
        result["next_ret"] = 0.0
        # day2 该组缺失，不能把 day3 与 day1 直接比较
        result["factor_quantile"] = [1, 1, np.nan, np.nan, 1, 1]

        group_turnover = f.calculate_group_turnover(result)
        self.assertEqual(group_turnover.index.tolist(), dates.tolist())
        self.assertAlmostEqual(group_turnover.loc[pd.Timestamp("2024-01-02"), "组1"], 1.0, places=10)
        self.assertAlmostEqual(group_turnover.loc[pd.Timestamp("2024-01-03"), "组1"], 1.0, places=10)

    def test_ic_stats_and_t_test_handle_near_constant_series(self):
        f = CrossSectionFactor("xs", self.data)
        near_const_ic = pd.Series(
            [0.1, 0.1 + 1e-12, 0.1 - 1e-12],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        ic_stats = f.ic_statistics_analysis(near_const_ic)
        self.assertEqual(ic_stats["IC标准差"], 0.0)
        self.assertEqual(ic_stats["IC信息比率(IR)"], 0.0)
        self.assertEqual(ic_stats["IC信息比率(IR,年化)"], 0.0)

        t_test = f.ic_mean_t_test(near_const_ic)
        self.assertTrue(np.isnan(t_test["T统计量"]))
        self.assertTrue(np.isnan(t_test["P值"]))
        self.assertEqual(t_test["显著性"], "方差过小")


if __name__ == "__main__":
    unittest.main()
