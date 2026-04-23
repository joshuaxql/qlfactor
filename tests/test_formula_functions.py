import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.factor_engine import Factor


class DummyFactor(Factor):
    def caculate(self) -> pd.Series:
        return self.CLOSE


class TestFormulaFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        symbols = ["AAA", "BBB"]

        raw = {
            "AAA": {
                "open": [1, 1, 2, 2, 4],
                "high": [1, 2, 3, 3, 5],
                "low": [1, 1, 2, 1, 4],
                "close": [1, 2, 3, 2, 5],
                "pre_close": [1, 1, 2, 3, 2],
                "change": [0, 1, 1, -1, 3],
                "pct_chg": [0, 100, 50, -33.3333, 150],
                "volume": [10, 20, 30, 40, 50],
                "amount": [100, 210, 330, 380, 700],
                "adj_factor": [1.0, 1.01, 1.02, 1.03, 1.04],
                "turnover_rate": [1, 2, 3, 4, 5],
                "turnover_rate_f": [2, 3, 4, 5, 6],
                "volume_ratio": [0.8, 0.9, 1.0, 1.1, 1.2],
                "pe": [10, 11, 12, 13, 14],
                "pe_ttm": [9, 10, 11, 12, 13],
                "pb": [1.0, 1.1, 1.2, 1.3, 1.4],
                "ps": [2.0, 2.1, 2.2, 2.3, 2.4],
                "ps_ttm": [1.8, 1.9, 2.0, 2.1, 2.2],
                "dv_ratio": [0.2, 0.2, 0.25, 0.25, 0.3],
                "dv_ttm": [0.3, 0.3, 0.35, 0.35, 0.4],
                "total_share": [1000, 1000, 1000, 1000, 1000],
                "float_share": [800, 800, 800, 800, 800],
                "free_share": [700, 700, 700, 700, 700],
                "total_mv": [10000, 10100, 10200, 10300, 10400],
                "circ_mv": [8000, 8080, 8160, 8240, 8320],
            },
            "BBB": {
                "open": [5, 4, 3, 4, 5],
                "high": [5, 5, 4, 5, 6],
                "low": [4, 3, 2, 3, 4],
                "close": [5, 4, 3, 4, 5],
                "pre_close": [5, 5, 4, 3, 4],
                "change": [0, -1, -1, 1, 1],
                "pct_chg": [0, -20, -25, 33.3333, 25],
                "volume": [50, 45, 40, 35, 30],
                "amount": [800, 650, 520, 610, 720],
                "adj_factor": [0.95, 0.96, 0.97, 0.98, 0.99],
                "turnover_rate": [5, 4, 3, 2, 1],
                "turnover_rate_f": [6, 5, 4, 3, 2],
                "volume_ratio": [1.2, 1.1, 1.0, 0.9, 0.8],
                "pe": [14, 13, 12, 11, 10],
                "pe_ttm": [13, 12, 11, 10, 9],
                "pb": [1.4, 1.3, 1.2, 1.1, 1.0],
                "ps": [2.4, 2.3, 2.2, 2.1, 2.0],
                "ps_ttm": [2.2, 2.1, 2.0, 1.9, 1.8],
                "dv_ratio": [0.3, 0.25, 0.25, 0.2, 0.2],
                "dv_ttm": [0.4, 0.35, 0.35, 0.3, 0.3],
                "total_share": [2000, 2000, 2000, 2000, 2000],
                "float_share": [1500, 1500, 1500, 1500, 1500],
                "free_share": [1400, 1400, 1400, 1400, 1400],
                "total_mv": [20000, 19900, 19800, 19700, 19600],
                "circ_mv": [15000, 14900, 14800, 14700, 14600],
            },
        }

        rows = []
        for dt_i, dt in enumerate(dates):
            for sym in symbols:
                row = {"date": dt, "symbol": sym}
                for col, values in raw[sym].items():
                    row[col] = values[dt_i]
                rows.append(row)

        df = pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
        cls.factor = DummyFactor("dummy", df)
        cls.data = df

    def test_basic_alias_columns(self):
        assert_series_equal(self.factor.CLOSE, self.data["close"])
        assert_series_equal(self.factor.OPEN, self.data["open"])
        assert_series_equal(self.factor.VOLUME, self.data["volume"])

    def test_all_added_alias_columns(self):
        alias_to_col = {
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "PRE_CLOSE": "pre_close",
            "CHANGE": "change",
            "PCT_CHG": "pct_chg",
            "VOLUME": "volume",
            "AMOUNT": "amount",
            "ADJ_FACTOR": "adj_factor",
            "TURNOVER_RATE": "turnover_rate",
            "TURNOVER_RATE_F": "turnover_rate_f",
            "VOLUME_RATIO": "volume_ratio",
            "PE": "pe",
            "PE_TTM": "pe_ttm",
            "PB": "pb",
            "PS": "ps",
            "PS_TTM": "ps_ttm",
            "DV_RATIO": "dv_ratio",
            "DV_TTM": "dv_ttm",
            "TOTAL_SHARE": "total_share",
            "FLOAT_SHARE": "float_share",
            "FREE_SHARE": "free_share",
            "TOTAL_MV": "total_mv",
            "CIRC_MV": "circ_mv",
        }
        for alias, col in alias_to_col.items():
            with self.subTest(alias=alias):
                assert_series_equal(getattr(self.factor, alias), self.data[col])

        expr = " + ".join(alias_to_col.keys())
        formula_sum = self.factor.FORMULA(expr)
        expected_sum = self.data[list(alias_to_col.values())].sum(axis=1)
        assert_series_equal(formula_sum, expected_sum)

    def test_ma_ema_rank(self):
        close = self.data["close"]
        ma = self.factor.MA(self.factor.CLOSE, 3)
        expected_ma = close.groupby(level="symbol").transform(
            lambda v: v.rolling(3, min_periods=3).mean()
        )
        assert_series_equal(ma, expected_ma)

        ema = self.factor.EMA(self.factor.CLOSE, 3, adjust=False)
        expected_ema = close.groupby(level="symbol").transform(
            lambda v: v.ewm(span=3, adjust=False, min_periods=1).mean()
        )
        assert_series_equal(ema, expected_ema)

        rank_ = self.factor.RANK(self.factor.CLOSE)
        expected_rank = close.groupby(level="date").rank(
            method="average", ascending=True, pct=True
        )
        assert_series_equal(rank_, expected_rank)

    def test_ref_delta_std_sum(self):
        close = self.data["close"]

        ref1 = self.factor.REF(close, 1)
        expected_ref1 = close.groupby(level="symbol").shift(1)
        assert_series_equal(ref1, expected_ref1)

        delta2 = self.factor.DELTA(close, 2)
        expected_delta2 = close.groupby(level="symbol").diff(2)
        assert_series_equal(delta2, expected_delta2)

        std3 = self.factor.STD(close, 3)
        expected_std3 = close.groupby(level="symbol").transform(
            lambda v: v.rolling(3, min_periods=3).std()
        )
        assert_series_equal(std3, expected_std3)

        sum3 = self.factor.SUM(close, 3)
        expected_sum3 = close.groupby(level="symbol").transform(
            lambda v: v.rolling(3, min_periods=3).sum()
        )
        assert_series_equal(sum3, expected_sum3)

    def test_tdx_style_scalar_and_logic(self):
        close = self.data["close"]
        ref1 = close.groupby(level="symbol").shift(1)
        cond = close > ref1

        abs_delta = self.factor.ABS(self.factor.DELTA(close, 1))
        expected_abs_delta = close.groupby(level="symbol").diff(1).abs()
        assert_series_equal(abs_delta, expected_abs_delta)

        max_s = self.factor.MAX(close, 4)
        expected_max = pd.concat(
            [close, pd.Series(4, index=self.data.index)], axis=1
        ).max(axis=1)
        assert_series_equal(max_s, expected_max)

        min_s = self.factor.MIN(close, 4)
        expected_min = pd.concat(
            [close, pd.Series(4, index=self.data.index)], axis=1
        ).min(axis=1)
        assert_series_equal(min_s, expected_min)

        if_s = self.factor.IF(cond, 1, 0)
        expected_if = pd.Series(np.where(cond.fillna(False), 1, 0), index=self.data.index)
        assert_series_equal(if_s, expected_if)

        count3 = self.factor.COUNT(cond, 3)
        expected_count3 = (
            cond.fillna(False)
            .astype(int)
            .groupby(level="symbol")
            .transform(lambda v: v.rolling(3, min_periods=3).sum())
        )
        assert_series_equal(count3, expected_count3)

    def test_tdx_style_window(self):
        close = self.data["close"]
        cond = close > close.groupby(level="symbol").shift(1)

        every3 = self.factor.EVERY(cond, 3)
        expected_every3 = (
            cond.fillna(False)
            .astype(int)
            .groupby(level="symbol")
            .transform(lambda v: v.rolling(3, min_periods=3).min())
            == 1
        )
        assert_series_equal(every3, expected_every3)

        exist3 = self.factor.EXIST(cond, 3)
        expected_exist3 = (
            cond.fillna(False)
            .astype(int)
            .groupby(level="symbol")
            .transform(lambda v: v.rolling(3, min_periods=3).max())
            == 1
        )
        assert_series_equal(exist3, expected_exist3)

        hhv3 = self.factor.HHV(close, 3)
        expected_hhv3 = close.groupby(level="symbol").transform(
            lambda v: v.rolling(3, min_periods=3).max()
        )
        assert_series_equal(hhv3, expected_hhv3)

        llv3 = self.factor.LLV(close, 3)
        expected_llv3 = close.groupby(level="symbol").transform(
            lambda v: v.rolling(3, min_periods=3).min()
        )
        assert_series_equal(llv3, expected_llv3)

        tsrank3 = self.factor.TSRANK(close, 3)
        expected_tsrank3 = close.groupby(level="symbol").transform(
            lambda v: v.rolling(3, min_periods=3).apply(
                lambda w: w.rank(pct=True).iloc[-1], raw=False
            )
        )
        assert_series_equal(tsrank3, expected_tsrank3)

    def test_sma_cross_formula_bind(self):
        close = self.data["close"]

        sma_3_1 = self.factor.SMA(close, 3, 1)
        expected_sma = close.groupby(level="symbol").transform(
            lambda v: v.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean()
        )
        assert_series_equal(sma_3_1, expected_sma)

        ma2 = self.factor.MA(close, 2, min_periods=2)
        ma3 = self.factor.MA(close, 3, min_periods=3)
        cross = self.factor.CROSS(ma2, ma3)
        expected_cross = (ma2 > ma3) & (
            ma2.groupby(level="symbol").shift(1) <= ma3.groupby(level="symbol").shift(1)
        )
        assert_series_equal(cross, expected_cross)

        formula = self.factor.FORMULA("MA(CLOSE, 2) - MA(CLOSE, 3)")
        expected_formula = self.factor.MA(close, 2) - self.factor.MA(close, 3)
        assert_series_equal(formula, expected_formula)

        MA, CLOSE = self.factor.BIND("MA", "CLOSE")
        bind_result = MA(CLOSE, 2)
        expected_bind = self.factor.MA(close, 2)
        assert_series_equal(bind_result, expected_bind)

    def test_manual_known_values(self):
        idx = pd.IndexSlice

        close_aaa = self.factor.CLOSE.xs("AAA", level="symbol", drop_level=False)

        ma3_aaa = self.factor.MA(self.factor.CLOSE, 3).loc[idx[:, "AAA"]]
        np.testing.assert_allclose(
            ma3_aaa.to_numpy(),
            np.array([np.nan, np.nan, 2.0, 7.0 / 3.0, 10.0 / 3.0]),
            equal_nan=True,
        )

        ema3_aaa = self.factor.EMA(self.factor.CLOSE, 3, adjust=False).loc[idx[:, "AAA"]]
        np.testing.assert_allclose(
            ema3_aaa.to_numpy(),
            np.array([1.0, 1.5, 2.25, 2.125, 3.5625]),
            atol=1e-10,
            equal_nan=True,
        )

        ref1_aaa = self.factor.REF(self.factor.CLOSE, 1).loc[idx[:, "AAA"]]
        np.testing.assert_allclose(
            ref1_aaa.to_numpy(), np.array([np.nan, 1.0, 2.0, 3.0, 2.0]), equal_nan=True
        )

        count3_aaa = self.factor.COUNT(
            self.factor.CLOSE > self.factor.REF(self.factor.CLOSE, 1), 3
        ).loc[idx[:, "AAA"]]
        np.testing.assert_allclose(
            count3_aaa.to_numpy(), np.array([np.nan, np.nan, 2.0, 2.0, 2.0]), equal_nan=True
        )

        hhv3_aaa = self.factor.HHV(self.factor.CLOSE, 3).loc[idx[:, "AAA"]]
        llv3_aaa = self.factor.LLV(self.factor.CLOSE, 3).loc[idx[:, "AAA"]]
        np.testing.assert_allclose(
            hhv3_aaa.to_numpy(), np.array([np.nan, np.nan, 3.0, 3.0, 5.0]), equal_nan=True
        )
        np.testing.assert_allclose(
            llv3_aaa.to_numpy(), np.array([np.nan, np.nan, 1.0, 2.0, 2.0]), equal_nan=True
        )

        tsrank3_aaa = self.factor.TSRANK(self.factor.CLOSE, 3).loc[idx[:, "AAA"]]
        np.testing.assert_allclose(
            tsrank3_aaa.to_numpy(), np.array([np.nan, np.nan, 1.0, 0.5, 1.0]), equal_nan=True
        )

        formula_aaa = self.factor.FORMULA("MA(CLOSE, 2) - MA(CLOSE, 3)").loc[idx[:, "AAA"]]
        np.testing.assert_allclose(
            formula_aaa.to_numpy(),
            np.array([np.nan, np.nan, 0.5, 1.0 / 6.0, 1.0 / 6.0]),
            atol=1e-10,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
