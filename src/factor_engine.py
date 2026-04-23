from abc import ABC, abstractmethod
import logging
import pandas as pd
from datetime import datetime
import numpy as np
from pyecharts.charts import Line, Bar, Page
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from scipy import stats


class Factor(ABC):
    def __init__(self, name: str, data: pd.DataFrame) -> None:
        """初始化因子。

        Args:
            name: 因子名称
            data: 股票日线数据，MultiIndex (date, symbol)，包含 close 等列
        """
        self.name = name
        self.data = data
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _series_stats(self, s: pd.Series) -> str:
        return (
            f"len={len(s)}, nan={int(s.isna().sum())}, "
            f"index_names={list(s.index.names)}"
        )

    def _df_stats(self, df: pd.DataFrame) -> str:
        return (
            f"shape={df.shape}, nan_total={int(df.isna().sum().sum())}, "
            f"columns={list(df.columns)}"
        )

    # ==================== 常用字段别名（公式风格） ====================
    @property
    def OPEN(self) -> pd.Series:
        return self.data["open"]

    @property
    def HIGH(self) -> pd.Series:
        return self.data["high"]

    @property
    def LOW(self) -> pd.Series:
        return self.data["low"]

    @property
    def CLOSE(self) -> pd.Series:
        return self.data["close"]

    @property
    def VOLUME(self) -> pd.Series:
        return self.data["volume"]

    @property
    def AMOUNT(self) -> pd.Series:
        return self.data["amount"]

    @property
    def TURNOVER_RATE(self) -> pd.Series:
        return self.data["turnover_rate"]

    @property
    def TURNOVER_RATE_F(self) -> pd.Series:
        return self.data["turnover_rate_f"]

    @property
    def PRE_CLOSE(self) -> pd.Series:
        return self.data["pre_close"]

    @property
    def CHANGE(self) -> pd.Series:
        return self.data["change"]

    @property
    def PCT_CHG(self) -> pd.Series:
        return self.data["pct_chg"]

    @property
    def ADJ_FACTOR(self) -> pd.Series:
        return self.data["adj_factor"]

    @property
    def VOLUME_RATIO(self) -> pd.Series:
        return self.data["volume_ratio"]

    @property
    def PE(self) -> pd.Series:
        return self.data["pe"]

    @property
    def PE_TTM(self) -> pd.Series:
        return self.data["pe_ttm"]

    @property
    def PB(self) -> pd.Series:
        return self.data["pb"]

    @property
    def PS(self) -> pd.Series:
        return self.data["ps"]

    @property
    def PS_TTM(self) -> pd.Series:
        return self.data["ps_ttm"]

    @property
    def DV_RATIO(self) -> pd.Series:
        return self.data["dv_ratio"]

    @property
    def DV_TTM(self) -> pd.Series:
        return self.data["dv_ttm"]

    @property
    def TOTAL_SHARE(self) -> pd.Series:
        return self.data["total_share"]

    @property
    def FLOAT_SHARE(self) -> pd.Series:
        return self.data["float_share"]

    @property
    def FREE_SHARE(self) -> pd.Series:
        return self.data["free_share"]

    @property
    def TOTAL_MV(self) -> pd.Series:
        return self.data["total_mv"]

    @property
    def CIRC_MV(self) -> pd.Series:
        return self.data["circ_mv"]

    def _resolve_series(self, x: str | pd.Series) -> pd.Series:
        """将字段名或Series统一解析为Series。"""
        if isinstance(x, pd.Series):
            return x
        if isinstance(x, str):
            key = x.lower()
            if key in self.data.columns:
                return self.data[key]
            raise KeyError(f"字段不存在: {x}")
        raise TypeError("参数必须是字段名(str)或pandas.Series")

    def _resolve_value(self, x: str | pd.Series | float | int | bool) -> pd.Series:
        """将字段名/Series/标量统一解析为与 data 对齐的 Series。"""
        if isinstance(x, (int, float, bool, np.number)):
            return pd.Series(x, index=self.data.index)

        s = self._resolve_series(x)
        if not s.index.equals(self.data.index):
            s = s.reindex(self.data.index)
        return s

    def _formula_namespace(self) -> dict[str, object]:
        """构造公式执行上下文（可直接使用 MA(CLOSE, 20) 风格）。"""
        return {
            # 字段
            "OPEN": self.OPEN,
            "HIGH": self.HIGH,
            "LOW": self.LOW,
            "CLOSE": self.CLOSE,
            "VOLUME": self.VOLUME,
            "AMOUNT": self.AMOUNT,
            "TURNOVER_RATE": self.TURNOVER_RATE,
            "TURNOVER_RATE_F": self.TURNOVER_RATE_F,
            "PRE_CLOSE": self.PRE_CLOSE,
            "CHANGE": self.CHANGE,
            "PCT_CHG": self.PCT_CHG,
            "ADJ_FACTOR": self.ADJ_FACTOR,
            "VOLUME_RATIO": self.VOLUME_RATIO,
            "PE": self.PE,
            "PE_TTM": self.PE_TTM,
            "PB": self.PB,
            "PS": self.PS,
            "PS_TTM": self.PS_TTM,
            "DV_RATIO": self.DV_RATIO,
            "DV_TTM": self.DV_TTM,
            "TOTAL_SHARE": self.TOTAL_SHARE,
            "FLOAT_SHARE": self.FLOAT_SHARE,
            "FREE_SHARE": self.FREE_SHARE,
            "TOTAL_MV": self.TOTAL_MV,
            "CIRC_MV": self.CIRC_MV,
            # 常用算子
            "MA": self.MA,
            "EMA": self.EMA,
            "RANK": self.RANK,
            "REF": self.REF,
            "DELTA": self.DELTA,
            "STD": self.STD,
            "SUM": self.SUM,
            # 通达信风格
            "ABS": self.ABS,
            "MAX": self.MAX,
            "MIN": self.MIN,
            "IF": self.IF,
            "COUNT": self.COUNT,
            "EVERY": self.EVERY,
            "EXIST": self.EXIST,
            "HHV": self.HHV,
            "LLV": self.LLV,
            "TSRANK": self.TSRANK,
            "SMA": self.SMA,
            "CROSS": self.CROSS,
            # 常用数值函数
            "LOG": np.log,
            "EXP": np.exp,
            "SQRT": np.sqrt,
        }

    def FORMULA(self, expr: str, **extra: object) -> pd.Series:
        """执行公式表达式，支持 MA(CLOSE, 20) 这类无 self 写法。"""
        ns = self._formula_namespace()
        ns.update(extra)
        self.logger.info("执行公式: %s", expr)
        try:
            result = eval(expr, {"__builtins__": {}}, ns)
            if isinstance(result, pd.Series):
                if not result.index.equals(self.data.index):
                    result = result.reindex(self.data.index)
                self.logger.info("公式执行完成: %s", self._series_stats(result))
                return result
            if isinstance(result, (int, float, bool, np.number)):
                out = pd.Series(result, index=self.data.index)
                self.logger.info("公式执行完成(标量扩展): %s", self._series_stats(out))
                return out
            raise TypeError("公式结果必须是 pandas.Series 或标量")
        except Exception:
            self.logger.exception("公式执行失败: %s", expr)
            raise

    def BIND(self, *names: str):
        """一次性绑定名称，减少重复写 self.。"""
        ns = self._formula_namespace()
        if not names:
            return ns
        values = []
        for name in names:
            if name not in ns:
                raise KeyError(f"未知名称: {name}")
            values.append(ns[name])
        return tuple(values)

    # ==================== 常用时序/截面算子 ====================
    def MA(
        self,
        x: str | pd.Series,
        n: int,
        min_periods: int | None = None,
    ) -> pd.Series:
        """移动平均，按 symbol 分组滚动计算。"""
        s = self._resolve_series(x)
        mp = n if min_periods is None else min_periods
        return s.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).mean()
        )

    def EMA(
        self,
        x: str | pd.Series,
        n: int,
        adjust: bool = True,
        min_periods: int = 1,
    ) -> pd.Series:
        """指数移动平均，按 symbol 分组计算。"""
        s = self._resolve_series(x)
        return s.groupby(level="symbol").transform(
            lambda v: v.ewm(span=n, adjust=adjust, min_periods=min_periods).mean()
        )

    def RANK(
        self,
        x: str | pd.Series,
        pct: bool = True,
        ascending: bool = True,
        method: str = "average",
    ) -> pd.Series:
        """截面排序，按 date 分组排名。"""
        s = self._resolve_series(x)
        return s.groupby(level="date").rank(method=method, ascending=ascending, pct=pct)

    def REF(self, x: str | pd.Series, n: int = 1) -> pd.Series:
        """滞后项，按 symbol 分组向后移动 n 期。"""
        s = self._resolve_series(x)
        return s.groupby(level="symbol").shift(n)

    def DELTA(self, x: str | pd.Series, n: int = 1) -> pd.Series:
        """差分，当前值减去 n 期前的值。"""
        s = self._resolve_series(x)
        return s.groupby(level="symbol").diff(n)

    def STD(
        self,
        x: str | pd.Series,
        n: int,
        min_periods: int | None = None,
    ) -> pd.Series:
        """滚动标准差，按 symbol 分组计算。"""
        s = self._resolve_series(x)
        mp = n if min_periods is None else min_periods
        return s.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).std()
        )

    def SUM(
        self,
        x: str | pd.Series,
        n: int,
        min_periods: int | None = None,
    ) -> pd.Series:
        """滚动求和，按 symbol 分组计算。"""
        s = self._resolve_series(x)
        mp = n if min_periods is None else min_periods
        return s.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).sum()
        )

    # ==================== 通达信风格常用函数 ====================
    def ABS(self, x: str | pd.Series) -> pd.Series:
        s = self._resolve_series(x)
        return s.abs()

    def MAX(
        self,
        a: str | pd.Series | float | int,
        b: str | pd.Series | float | int,
    ) -> pd.Series:
        sa = self._resolve_value(a)
        sb = self._resolve_value(b)
        return pd.concat([sa, sb], axis=1).max(axis=1)

    def MIN(
        self,
        a: str | pd.Series | float | int,
        b: str | pd.Series | float | int,
    ) -> pd.Series:
        sa = self._resolve_value(a)
        sb = self._resolve_value(b)
        return pd.concat([sa, sb], axis=1).min(axis=1)

    def IF(
        self,
        cond: str | pd.Series | float | int | bool,
        a: str | pd.Series | float | int,
        b: str | pd.Series | float | int,
    ) -> pd.Series:
        c = self._resolve_value(cond).fillna(False).astype(bool)
        sa = self._resolve_value(a)
        sb = self._resolve_value(b)
        return pd.Series(np.where(c, sa, sb), index=self.data.index)

    def COUNT(
        self,
        cond: str | pd.Series | float | int | bool,
        n: int,
        min_periods: int | None = None,
    ) -> pd.Series:
        c = self._resolve_value(cond).fillna(False).astype(bool).astype(int)
        mp = n if min_periods is None else min_periods
        return c.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).sum()
        )

    def EVERY(
        self,
        cond: str | pd.Series | float | int | bool,
        n: int,
        min_periods: int | None = None,
    ) -> pd.Series:
        """最近 n 期是否全部满足条件。"""
        c = self._resolve_value(cond).fillna(False).astype(bool).astype(int)
        mp = n if min_periods is None else min_periods
        out = c.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).min()
        )
        return out == 1

    def EXIST(
        self,
        cond: str | pd.Series | float | int | bool,
        n: int,
        min_periods: int | None = None,
    ) -> pd.Series:
        """最近 n 期是否至少一次满足条件。"""
        c = self._resolve_value(cond).fillna(False).astype(bool).astype(int)
        mp = n if min_periods is None else min_periods
        out = c.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).max()
        )
        return out == 1

    def HHV(
        self,
        x: str | pd.Series,
        n: int,
        min_periods: int | None = None,
    ) -> pd.Series:
        """最近 n 期最高值（按 symbol 滚动）。"""
        s = self._resolve_series(x)
        mp = n if min_periods is None else min_periods
        return s.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).max()
        )

    def LLV(
        self,
        x: str | pd.Series,
        n: int,
        min_periods: int | None = None,
    ) -> pd.Series:
        """最近 n 期最低值（按 symbol 滚动）。"""
        s = self._resolve_series(x)
        mp = n if min_periods is None else min_periods
        return s.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).min()
        )

    def TSRANK(
        self,
        x: str | pd.Series,
        n: int,
        pct: bool = True,
        min_periods: int | None = None,
    ) -> pd.Series:
        """时序排序：当前值在最近 n 期窗口内的排名。"""
        s = self._resolve_series(x)
        mp = n if min_periods is None else min_periods

        def _rank_last(window: pd.Series) -> float:
            return window.rank(pct=pct).iloc[-1]

        return s.groupby(level="symbol").transform(
            lambda v: v.rolling(n, min_periods=mp).apply(_rank_last, raw=False)
        )

    def SMA(
        self,
        x: str | pd.Series,
        n: int,
        m: int = 1,
        min_periods: int = 1,
    ) -> pd.Series:
        """通达信SMA：Y=(M*X+(N-M)*Y')/N。"""
        if n <= 0:
            raise ValueError("n 必须大于 0")
        if not (0 < m <= n):
            raise ValueError("m 必须满足 0 < m <= n")
        s = self._resolve_series(x)
        alpha = m / n
        return s.groupby(level="symbol").transform(
            lambda v: v.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()
        )

    def CROSS(
        self,
        a: str | pd.Series | float | int,
        b: str | pd.Series | float | int,
    ) -> pd.Series:
        """上穿信号：本期 a>b 且上期 a<=b。"""
        sa = self._resolve_value(a)
        sb = self._resolve_value(b)
        prev_a = sa.groupby(level="symbol").shift(1)
        prev_b = sb.groupby(level="symbol").shift(1)
        return (sa > sb) & (prev_a <= prev_b)

    @abstractmethod
    def caculate(self) -> pd.Series:
        """计算因子值，必须由子类实现。

        Returns:
            因子值 Series，索引为 (date, symbol)
        """
        ...

    def winsorize_3sigma(self, factor: pd.Series, n: int = 3) -> pd.Series:
        """3-sigma法去极值：超出 n 倍标准差的 values 替换为边界值。

        Args:
            factor: 因子值序列
            n: 标准差倍数阈值（默认3）

        Returns:
            去极值后的因子 Series
        """
        mean = factor.mean()
        std = factor.std()
        if pd.isna(std) or std == 0:
            return factor
        lower = mean - n * std
        upper = mean + n * std
        return factor.clip(lower=lower, upper=upper)

    def winsorize_mad(self, factor: pd.Series, k: float = 3) -> pd.Series:
        """MAD法去极值：中位数绝对偏差法，超出 k*MAD 的 values 替换为边界值。

        Args:
            factor: 因子值序列
            k: MAD倍数阈值（默认3）

        Returns:
            去极值后的因子 Series
        """
        median = factor.median()
        mad = (factor - median).abs().median()
        if mad == 0:
            return factor
        lower = median - k * mad
        upper = median + k * mad
        return factor.clip(lower=lower, upper=upper)

    def standardize_zscore(self, factor: pd.Series) -> pd.Series:
        """Z-score标准化：(value - mean) / std，输出均值为0，标准差为1。

        Args:
            factor: 因子值序列

        Returns:
            标准化后的因子 Series
        """
        std = factor.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=factor.index)
        return (factor - factor.mean()) / std

    def _prepare_factor_values(
        self,
        winsorize: str | None = None,
        winsorize_param: float = 3,
        standardize: bool = False,
    ) -> pd.Series:
        """对原始因子值做去极值和标准化处理。"""
        self.logger.info(
            "开始计算因子值: name=%s, winsorize=%s, winsorize_param=%s, standardize=%s",
            self.name,
            winsorize,
            winsorize_param,
            standardize,
        )
        try:
            factor = self.caculate()
            self.logger.info("原始因子统计: %s", self._series_stats(factor))

            if winsorize == "3sigma":
                self.logger.info("执行去极值: 3sigma")
                factor = factor.groupby(level="date").transform(
                    lambda x: self.winsorize_3sigma(x, winsorize_param)
                )
            elif winsorize == "mad":
                self.logger.info("执行去极值: mad")
                factor = factor.groupby(level="date").transform(
                    lambda x: self.winsorize_mad(x, winsorize_param)
                )

            if standardize:
                self.logger.info("执行标准化: zscore")
                factor = factor.groupby(level="date").transform(self.standardize_zscore)

            self.logger.info("处理后因子统计: %s", self._series_stats(factor))
            return factor
        except Exception:
            self.logger.exception("因子值计算失败: %s", self.name)
            raise

    def get_clean_factor_and_forward_returns(
        self,
        quantiles: int = 5,
        period: int = 1,
        winsorize: str | None = None,
        winsorize_param: float = 3,
        standardize: bool = False,
    ) -> pd.DataFrame:
        """清洗因子数据并计算多周期远期收益，生成因子分析标准数据集。

        Args:
            quantiles: 因子每日截面分组数（按因子值分组，如分5组）
            period: 远期收益计算周期（天数）
            winsorize: 去极值方法，"3sigma" 或 "mad"，None则不做去极值
            winsorize_param: 去极值参数，3sigma为标准差倍数(默认3)，mad为MAD倍数(默认3)
            standardize: 是否做Z-score标准化

        Returns:
            干净的长格式 DataFrame，列包含：
            date, symbol, factor, next_ret, factor_quantile
        """
        # 1. 获取并预处理因子值
        factor = self._prepare_factor_values(winsorize, winsorize_param, standardize)

        # 4. 获取价格数据用于计算远期收益
        price = self.data["close"]

        # 5. 构建结果DataFrame（对齐索引）
        result = pd.DataFrame({"factor": factor, "close": price})

        # 6. 计算远期收益
        ret = result["close"].groupby(level="symbol").pct_change(period)
        result["next_ret"] = ret.groupby(level="symbol").shift(-period)

        # 7. 清洗：去除NaN
        result = result.dropna()

        # 8. 计算因子分位数（每日截面）
        result["factor_quantile"] = result.groupby(level="date")["factor"].transform(
            lambda x: pd.qcut(x, q=quantiles, labels=False, duplicates="drop") + 1
        )

        return result

    def caculate_daily_ic(self, result: pd.DataFrame) -> pd.Series:
        """计算每日 IC（Rank IC，Spearman 相关系数）。

        Args:
            result: get_clean_factor_and_forward_returns 返回的 DataFrame

        Returns:
            每日 IC Series，索引为 date
        """

        def _safe_spearman(x: pd.DataFrame) -> float:
            fac = x["factor"]
            ret = x["next_ret"]
            if fac.nunique(dropna=True) < 2 or ret.nunique(dropna=True) < 2:
                return np.nan
            return fac.corr(ret, method="spearman")

        daily_ic = result.groupby(level="date").apply(_safe_spearman)
        daily_ic.name = "IC"

        return daily_ic

    def ic_statistics_analysis(self, daily_ic: pd.Series) -> pd.Series:
        """计算 IC 统计指标：IC均值、标准差、信息比率（IR）。

        Args:
            daily_ic: 每日 IC 序列

        Returns:
            包含 IC均值、IC标准差、IC信息比率的 Series
        """
        ic_analysis = pd.Series(
            {
                "IC均值": round(daily_ic.mean(), 4),
                "IC标准差": round(daily_ic.std(), 4),
                "IC信息比率(IR)": (
                    round(daily_ic.mean() / daily_ic.std(), 4)
                    if daily_ic.std() != 0
                    else 0
                ),
                "IC信息比率(IR,年化)": (
                    round(daily_ic.mean() / daily_ic.std() * np.sqrt(252), 4)
                    if daily_ic.std() != 0
                    else 0
                ),
            }
        )
        return ic_analysis

    def ic_mean_t_test(self, daily_ic: pd.Series) -> pd.Series:
        """对每日 IC 序列进行单样本 T 检验，检验 IC 均值是否显著不为零。

        Args:
            daily_ic: 每日 IC 序列

        Returns:
            包含 T统计量、P值、显著性判断的 Series
        """
        # 剔除空值
        ic = daily_ic.dropna()
        if len(ic) < 2:
            return pd.Series({"T统计量": np.nan, "P值": np.nan, "显著性": "样本不足"})
        # 单样本T检验（对比值=0）
        t_stat, p_value = stats.ttest_1samp(ic, popmean=0)
        # 显著性判断
        significant = "显著（p<0.05）" if p_value < 0.05 else "不显著（p≥0.05）"

        result = pd.Series(
            {
                "T统计量": round(t_stat, 4),
                "P值": round(p_value, 4),
                "显著性": significant,
            }
        )
        return result

    def plot_ic_time_series(self, daily_ic: pd.Series) -> Line:
        """绘制因子每日 IC 时序图。

        Args:
            daily_ic: 每日 IC 序列

        Returns:
            Pyecharts Line 图表对象
        """
        x_data = daily_ic.index.strftime("%Y-%m-%d").tolist()
        y_data = daily_ic.round(4).tolist()

        line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        line.add_xaxis(xaxis_data=x_data)

        line.add_yaxis(
            series_name="每日IC值",
            y_axis=y_data,
            is_smooth=False,
            symbol=None,  # 关闭数据圆点
            label_opts=opts.LabelOpts(is_show=False),  # 关闭数值显示
            linestyle_opts=opts.LineStyleOpts(width=2, color="#2E86AB"),
        )

        line.add_yaxis(
            series_name="IC平均值",
            y_axis=[round(daily_ic.mean(), 4)] * len(x_data),
            is_smooth=False,
            symbol=None,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=1, color="red"),
        )

        line.set_global_opts(
            title_opts=opts.TitleOpts(title="因子每日IC时序图"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(name="IC值"),
            legend_opts=opts.LegendOpts(pos_left="center", pos_top="7%"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        )

        return line

    def plot_ic_cumulative(self, daily_ic: pd.Series) -> Line:
        """绘制 IC 累加曲线，观察因子持续有效性。

        Args:
            daily_ic: 每日 IC 序列

        Returns:
            Pyecharts Line 图表对象
        """
        x_data = daily_ic.index.strftime("%Y-%m-%d").tolist()
        y_cum_ic = daily_ic.cumsum().round(4).tolist()  # 计算累加IC

        line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        line.add_xaxis(xaxis_data=x_data)

        line.add_yaxis(
            series_name="IC累加曲线",
            y_axis=y_cum_ic,
            is_smooth=False,
            symbol=None,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#FF6B6B"),
        )

        line.set_global_opts(
            title_opts=opts.TitleOpts(title="因子IC累加曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(name="累计IC"),
            legend_opts=opts.LegendOpts(pos_left="center", pos_top="7%"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        )

        return line

    def calculate_daily_group_ret(self, result: pd.DataFrame) -> pd.DataFrame:
        """按日期和因子分组计算每日等权平均收益。

        Args:
            result: get_clean_factor_and_forward_returns 返回的 DataFrame

        Returns:
            日期为行索引、分组为列的每日收益 DataFrame
        """
        group_daily = (
            result.groupby(["date", "factor_quantile"])["next_ret"].mean().reset_index()
        )
        group_ret = group_daily.pivot(
            index="date", columns="factor_quantile", values="next_ret"
        )
        group_ret.columns = [f"组{int(i)}" for i in group_ret.columns]
        return group_ret

    def calculate_group_turnover(self, result: pd.DataFrame) -> pd.DataFrame:
        """计算各因子分组的日度持仓换手率。

        换手率定义为：1 - 前后两日持仓重合比例。

        Args:
            result: get_clean_factor_and_forward_returns 返回的 DataFrame

        Returns:
            日期为行索引、分组为列的换手率 DataFrame
        """
        members = (
            result.reset_index()[["date", "symbol", "factor_quantile"]]
            .dropna(subset=["factor_quantile"])
            .copy()
        )
        quantiles = sorted(members["factor_quantile"].unique())

        turnover_map: dict[str, pd.Series] = {}
        for q in quantiles:
            by_date = (
                members[members["factor_quantile"] == q]
                .groupby("date")["symbol"]
                .apply(set)
                .sort_index()
            )

            prev_set: set[str] | None = None
            values: list[float] = []
            for cur_set in by_date:
                if prev_set is None or len(prev_set) == 0:
                    values.append(np.nan)
                else:
                    overlap = len(cur_set & prev_set)
                    values.append(1 - overlap / len(prev_set))
                prev_set = cur_set

            turnover_map[f"组{int(q)}"] = pd.Series(values, index=by_date.index)

        return pd.DataFrame(turnover_map).sort_index()

    def calculate_group_turnover_stats(
        self, group_turnover: pd.DataFrame
    ) -> pd.DataFrame:
        """汇总分组换手率统计。"""
        return pd.DataFrame(
            {
                "平均换手率": group_turnover.mean(),
                "换手率标准差": group_turnover.std(),
                "P25": group_turnover.quantile(0.25),
                "P50": group_turnover.quantile(0.50),
                "P75": group_turnover.quantile(0.75),
            }
        ).round(4)

    def calculate_factor_turnover_rate(self, factor: pd.Series) -> pd.Series:
        """计算因子整体换手率（1 - 截面秩自相关）。

        Args:
            factor: 预处理后的因子值，索引为 (date, symbol)

        Returns:
            因子换手率序列，索引为 date
        """
        factor_wide = factor.unstack("symbol").sort_index()
        factor_rank = factor_wide.rank(axis=1, pct=True)

        rank_autocorr = pd.Series(index=factor_rank.index, dtype=float)
        prev = factor_rank.shift(1)
        for dt in factor_rank.index:
            cur_row = factor_rank.loc[dt]
            prev_row = prev.loc[dt]
            pair = pd.concat([cur_row, prev_row], axis=1, keys=["cur", "prev"]).dropna()
            if pair.empty or pair["cur"].nunique() < 2 or pair["prev"].nunique() < 2:
                rank_autocorr.loc[dt] = np.nan
            else:
                rank_autocorr.loc[dt] = pair["cur"].corr(
                    pair["prev"], method="spearman"
                )

        turnover = 1 - rank_autocorr
        turnover.name = "因子换手率"
        return turnover

    def factor_turnover_stats(self, factor_turnover: pd.Series) -> pd.Series:
        """汇总因子换手率统计。"""
        return pd.Series(
            {
                "平均换手率": round(float(factor_turnover.mean()), 4),
                "换手率标准差": round(float(factor_turnover.std()), 4),
                "P25": round(float(factor_turnover.quantile(0.25)), 4),
                "P50": round(float(factor_turnover.quantile(0.50)), 4),
                "P75": round(float(factor_turnover.quantile(0.75)), 4),
            }
        )

    def calculate_turnover_return_correlation(
        self, group_turnover: pd.DataFrame, group_ret: pd.DataFrame
    ) -> pd.Series:
        """计算各分组换手率与分组收益的相关系数。"""
        aligned_turnover, aligned_ret = group_turnover.align(
            group_ret, join="inner", axis=0
        )
        common_cols = aligned_turnover.columns.intersection(aligned_ret.columns)

        corr_dict: dict[str, float] = {}
        for col in common_cols:
            pair = pd.concat(
                [aligned_turnover[col], aligned_ret[col]],
                axis=1,
                keys=["turnover", "ret"],
            ).dropna()
            if len(pair) < 3:
                corr_dict[col] = np.nan
            else:
                corr_dict[col] = pair["turnover"].corr(pair["ret"])
        return pd.Series(corr_dict, name="换手率收益相关系数").round(4)

    def calculate_net_returns_with_cost(
        self,
        group_ret: pd.DataFrame,
        group_turnover: pd.DataFrame,
        cost_bps: float = 10,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """按换手率扣除交易成本后，计算分组与多空净收益。

        Args:
            group_ret: 每日分组收益
            group_turnover: 每日分组换手率
            cost_bps: 单边交易成本（bps）

        Returns:
            (分组净收益, 多空毛收益, 多空净收益, 多空净绩效)
        """
        cost_rate = cost_bps / 10000
        aligned_turnover, aligned_ret = group_turnover.align(
            group_ret, join="inner", axis=0
        )
        common_cols = aligned_turnover.columns.intersection(aligned_ret.columns)

        group_net_ret = (
            aligned_ret[common_cols] - aligned_turnover[common_cols] * cost_rate
        )

        ordered_cols = sorted(
            common_cols, key=lambda x: float(str(x).replace("组", ""))
        )
        low_col = ordered_cols[0]
        high_col = ordered_cols[-1]

        long_short_gross = aligned_ret[high_col] - aligned_ret[low_col]
        long_short_net = (
            long_short_gross
            - (aligned_turnover[high_col] + aligned_turnover[low_col]) * cost_rate
        )
        long_short_net_perf = self.performance_analysis(long_short_net)
        long_short_net_perf.name = f"多空净收益({high_col}-{low_col}, {cost_bps}bps)"

        return group_net_ret, long_short_gross, long_short_net, long_short_net_perf

    def performance_analysis(self, ret_series: pd.Series, period: int = 1) -> pd.Series:
        """计算单组收益的核心量化指标。

        Args:
            ret_series: 日收益序列
            period: 收益对应的持有周期（天）

        Returns:
            包含年化收益、年化波动率、夏普比率、总累计收益的 Series
        """
        annual_factor = 252 / max(period, 1)
        ann_ret = ret_series.mean() * annual_factor
        ann_vol = ret_series.std() * np.sqrt(annual_factor)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0  # 夏普比率
        cum_ret = (1 + ret_series).prod() - 1  # 总累计收益
        return pd.Series(
            {
                "年化收益": round(ann_ret, 4),
                "年化波动率": round(ann_vol, 4),
                "夏普比率": round(sharpe, 4),
                "总累计收益": round(cum_ret, 4),
            }
        )

    def calculate_all_group_performance(
        self, group_ret: pd.DataFrame, period: int = 1
    ) -> pd.DataFrame:
        """计算所有因子分组的绩效指标汇总表。

        Args:
            group_ret: calculate_daily_group_ret 返回的每日分组收益表
            period: 收益对应的持有周期（天）

        Returns:
            分组绩效汇总 DataFrame
        """
        return group_ret.apply(lambda x: self.performance_analysis(x, period)).T

    def calculate_long_short(
        self, group_ret: pd.DataFrame, period: int = 1
    ) -> tuple[pd.Series, pd.Series]:
        """计算多空组合收益（最高因子组 - 最低因子组）。

        Args:
            group_ret: calculate_daily_group_ret 返回的每日分组收益表
            period: 收益对应的持有周期（天）

        Returns:
            (多空日收益 Series, 多空绩效指标 Series)
        """
        ordered_cols = sorted(
            group_ret.columns, key=lambda x: float(str(x).replace("组", ""))
        )
        low_col = ordered_cols[0]
        high_col = ordered_cols[-1]

        long_short_ret = group_ret[high_col] - group_ret[low_col]
        long_short_perf = self.performance_analysis(long_short_ret, period)
        long_short_perf.name = f"多空组合({high_col}-{low_col})"
        return long_short_ret, long_short_perf

    def plot_group_annual_bar(self, group_perf_df: pd.DataFrame) -> Bar:
        """绘制因子分组年化收益对比柱状图。

        Args:
            group_perf_df: calculate_all_group_performance 返回的分组绩效表

        Returns:
            Pyecharts Bar 图表对象
        """
        bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        bar.add_xaxis(group_perf_df.index.tolist())
        bar.add_yaxis(
            "年化收益",
            group_perf_df["年化收益"].tolist(),
            label_opts=opts.LabelOpts(is_show=False),
        )
        bar.set_global_opts(
            title_opts=opts.TitleOpts(title="因子分组年化收益对比"),
            xaxis_opts=opts.AxisOpts(name="因子分组"),
            yaxis_opts=opts.AxisOpts(name="年化收益率"),
        )
        return bar

    def plot_ic_histogram(self, daily_ic: pd.Series) -> Bar:
        """绘制 IC 分布直方图。

        Args:
            daily_ic: 每日 IC 序列

        Returns:
            Pyecharts Bar 图表对象
        """
        bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        clean_ic = daily_ic.dropna()
        if clean_ic.empty:
            bar.add_xaxis(["无有效IC数据"])
            bar.add_yaxis("IC频次", [0], label_opts=opts.LabelOpts(is_show=False))
        else:
            hist, bins = np.histogram(clean_ic, bins=30)
            bar.add_xaxis(
                [round((bins[i] + bins[i + 1]) / 2, 3) for i in range(len(bins) - 1)]
            )
            bar.add_yaxis(
                "IC频次", hist.tolist(), label_opts=opts.LabelOpts(is_show=False)
            )
        bar.set_global_opts(
            title_opts=opts.TitleOpts(title="IC分布直方图"),
            xaxis_opts=opts.AxisOpts(name="IC值"),
            yaxis_opts=opts.AxisOpts(name="频次"),
        )
        return bar

    def plot_group_cumulative_return(
        self, group_ret: pd.DataFrame, long_short_ret: pd.Series
    ) -> Line:
        """绘制因子分组累计收益曲线及多空组合净值曲线。

        Args:
            group_ret: calculate_daily_group_ret 返回的每日分组收益表
            long_short_ret: calculate_long_short 返回的多空日收益序列

        Returns:
            Pyecharts Line 图表对象
        """
        x_data = group_ret.index.strftime("%Y-%m-%d").tolist()
        group_ret = (1 + group_ret).cumprod().round(4)
        ls_cum = (1 + long_short_ret).cumprod().round(4).tolist()

        line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        line.add_xaxis(xaxis_data=x_data)

        for col in group_ret.columns:
            line.add_yaxis(
                series_name=col,
                y_axis=group_ret[col].tolist(),
                symbol=None,
                label_opts=opts.LabelOpts(is_show=False),
            )

        ordered_cols = sorted(
            group_ret.columns, key=lambda x: float(str(x).replace("组", ""))
        )
        ls_name = f"多空组合({ordered_cols[-1]}-{ordered_cols[0]})"

        line.add_yaxis(
            series_name=ls_name,
            y_axis=ls_cum,
            symbol=None,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=3, type_="dashed", color="black"),
        )

        line.set_global_opts(
            title_opts=opts.TitleOpts(title="因子分组累计收益曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),  # 鼠标悬浮显示数据
            legend_opts=opts.LegendOpts(pos_left="center", pos_top="7%"),
            datazoom_opts=[
                opts.DataZoomOpts(type_="slider"),  # 底部滑动缩放
                opts.DataZoomOpts(type_="inside"),  # 鼠标滚轮缩放
            ],
        )

        return line

    def plot_group_turnover(self, group_turnover: pd.DataFrame) -> Line:
        """绘制分组换手率时序图。"""
        x_data = group_turnover.index.strftime("%Y-%m-%d").tolist()

        line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        line.add_xaxis(xaxis_data=x_data)

        ordered_cols = sorted(
            group_turnover.columns, key=lambda x: float(str(x).replace("组", ""))
        )
        for col in ordered_cols:
            line.add_yaxis(
                series_name=col,
                y_axis=group_turnover[col].round(4).tolist(),
                symbol=None,
                label_opts=opts.LabelOpts(is_show=False),
            )

        line.set_global_opts(
            title_opts=opts.TitleOpts(title="因子分组换手率时序图"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(name="换手率"),
            legend_opts=opts.LegendOpts(pos_left="center", pos_top="7%"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        )
        return line

    def plot_group_turnover_bar(self, turnover_stats: pd.DataFrame) -> Bar:
        """绘制分组平均换手率柱状图。"""
        bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        bar.add_xaxis(turnover_stats.index.tolist())
        bar.add_yaxis(
            "平均换手率",
            turnover_stats["平均换手率"].tolist(),
            label_opts=opts.LabelOpts(is_show=False),
        )
        bar.set_global_opts(
            title_opts=opts.TitleOpts(title="因子分组平均换手率"),
            xaxis_opts=opts.AxisOpts(name="因子分组"),
            yaxis_opts=opts.AxisOpts(name="平均换手率"),
        )
        return bar

    def plot_factor_turnover(self, factor_turnover: pd.Series) -> Line:
        """绘制因子整体换手率时序图。"""
        clean_turnover = factor_turnover.dropna()
        x_data = clean_turnover.index.strftime("%Y-%m-%d").tolist()
        y_data = clean_turnover.round(4).tolist()

        line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        line.add_xaxis(xaxis_data=x_data)
        line.add_yaxis(
            series_name="因子换手率",
            y_axis=y_data,
            symbol=None,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#1F77B4"),
        )
        if len(y_data) > 0:
            line.add_yaxis(
                series_name="平均换手率",
                y_axis=[round(float(clean_turnover.mean()), 4)] * len(y_data),
                symbol=None,
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=1, color="red", type_="dashed"),
            )

        line.set_global_opts(
            title_opts=opts.TitleOpts(title="因子整体换手率时序图"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(name="换手率"),
            legend_opts=opts.LegendOpts(pos_left="center", pos_top="7%"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        )
        return line

    def plot_turnover_return_correlation(self, corr_series: pd.Series) -> Bar:
        """绘制分组换手率与收益相关系数柱状图。"""
        bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        if corr_series.empty:
            bar.add_xaxis(["无有效数据"])
            bar.add_yaxis("相关系数", [0], label_opts=opts.LabelOpts(is_show=False))
        else:
            bar.add_xaxis(corr_series.index.tolist())
            bar.add_yaxis(
                "相关系数",
                corr_series.fillna(0).tolist(),
                label_opts=opts.LabelOpts(is_show=False),
            )

        bar.set_global_opts(
            title_opts=opts.TitleOpts(title="分组换手率与收益相关系数"),
            xaxis_opts=opts.AxisOpts(name="因子分组"),
            yaxis_opts=opts.AxisOpts(name="相关系数", min_=-1, max_=1),
        )
        return bar

    def plot_long_short_net_vs_gross(
        self,
        long_short_gross: pd.Series,
        long_short_net: pd.Series,
        cost_bps: float,
    ) -> Line:
        """绘制多空毛收益与净收益累计曲线。"""
        aligned = pd.concat(
            [long_short_gross, long_short_net],
            axis=1,
            keys=["毛收益", "净收益"],
        ).dropna()

        gross_cum = (1 + aligned["毛收益"]).cumprod().round(4)
        net_cum = (1 + aligned["净收益"]).cumprod().round(4)
        cost_drag = (gross_cum - net_cum).round(4)

        x_data = aligned.index.strftime("%Y-%m-%d").tolist()
        line = Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        line.add_xaxis(xaxis_data=x_data)
        line.add_yaxis(
            series_name="多空毛收益累计",
            y_axis=gross_cum.tolist(),
            symbol=None,
            label_opts=opts.LabelOpts(is_show=False),
        )
        line.add_yaxis(
            series_name=f"多空净收益累计({cost_bps}bps)",
            y_axis=net_cum.tolist(),
            symbol=None,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=2, color="#D9534F"),
        )
        line.add_yaxis(
            series_name="累计成本拖累",
            y_axis=cost_drag.tolist(),
            symbol=None,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=1, type_="dashed", color="black"),
        )
        line.set_global_opts(
            title_opts=opts.TitleOpts(title="多空毛净收益对比曲线"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_left="center", pos_top="7%"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        )
        return line

    def create_factor_analysis_report(
        self,
        quantiles: int = 5,
        period: int = 1,
        winsorize: str | None = None,
        winsorize_param: float = 3,
        standardize: bool = False,
        transaction_cost_bps: float = 10,
    ) -> None:
        """生成完整的因子分析报告，包含 IC 时序、分组收益、多空组合等图表。

        Args:
            quantiles: 因子每日截面分组数（默认5）
            period: 远期收益计算周期天数（默认1）
            winsorize: 去极值方法，"3sigma" 或 "mad"，None 跳过
            winsorize_param: 去极值参数（默认3）
            standardize: 是否做 Z-score 标准化（默认 False）
            transaction_cost_bps: 单边交易成本（bps，默认10）
        """
        self.logger.info(
            "开始生成因子报告: name=%s, quantiles=%s, period=%s, winsorize=%s, standardize=%s, cost_bps=%s",
            self.name,
            quantiles,
            period,
            winsorize,
            standardize,
            transaction_cost_bps,
        )
        try:
            result = self.get_clean_factor_and_forward_returns(
                quantiles, period, winsorize, winsorize_param, standardize
            )
            self.logger.info("clean因子数据: %s", self._df_stats(result))

            daily_group_ret = self.calculate_daily_group_ret(result)
            self.logger.info("分组收益矩阵: %s", self._df_stats(daily_group_ret))

            group_perf = self.calculate_all_group_performance(daily_group_ret, period)
            ls_ret, ls_perf = self.calculate_long_short(daily_group_ret, period)
            daily_ic = self.caculate_daily_ic(result)
            self.logger.info("IC序列: %s", self._series_stats(daily_ic))

            ic_stats = self.ic_statistics_analysis(daily_ic)
            ic_t = self.ic_mean_t_test(daily_ic)
            group_turnover = self.calculate_group_turnover(result)
            self.logger.info("分组换手率: %s", self._df_stats(group_turnover))

            group_turnover_stats = self.calculate_group_turnover_stats(group_turnover)
            factor_values = self._prepare_factor_values(
                winsorize=winsorize,
                winsorize_param=winsorize_param,
                standardize=standardize,
            ).dropna()
            self.logger.info(
                "用于换手率的因子值: %s", self._series_stats(factor_values)
            )

            factor_turnover = self.calculate_factor_turnover_rate(factor_values)
            factor_turnover_summary = self.factor_turnover_stats(factor_turnover)
            turnover_ret_corr = self.calculate_turnover_return_correlation(
                group_turnover, daily_group_ret
            )
            group_net_ret, ls_gross_ret, ls_net_ret, ls_net_perf = (
                self.calculate_net_returns_with_cost(
                    daily_group_ret,
                    group_turnover,
                    cost_bps=transaction_cost_bps,
                )
            )
            group_net_perf = self.calculate_all_group_performance(group_net_ret, period)

            page = Page(page_title="因子分析报告", layout=Page.SimplePageLayout)
            page.add(self.plot_group_cumulative_return(daily_group_ret, ls_ret))
            page.add(self.plot_ic_time_series(daily_ic))
            page.add(self.plot_ic_cumulative(daily_ic))
            page.add(self.plot_group_annual_bar(group_perf))
            page.add(self.plot_ic_histogram(daily_ic))
            page.add(self.plot_group_turnover(group_turnover))
            page.add(self.plot_group_turnover_bar(group_turnover_stats))
            page.add(self.plot_factor_turnover(factor_turnover))
            page.add(self.plot_turnover_return_correlation(turnover_ret_corr))
            page.add(
                self.plot_long_short_net_vs_gross(
                    ls_gross_ret, ls_net_ret, cost_bps=transaction_cost_bps
                )
            )
            page.render("./output/因子分析报告.html")
            self.logger.info("报告已写入: ./output/因子分析报告.html")

            print("\n分组绩效指标\n", group_perf)
            print("\n多空组合指标\n", ls_perf)
            print("\nIC统计指标\n", ic_stats)
            print("\nIC T检验\n", ic_t)
            print("\n分组换手率统计\n", group_turnover_stats)
            print("\n因子换手率统计\n", factor_turnover_summary)
            print("\n换手率与收益相关系数\n", turnover_ret_corr)
            print(f"\n分组净绩效指标(扣{transaction_cost_bps}bps)\n", group_net_perf)
            print(f"\n多空净绩效指标(扣{transaction_cost_bps}bps)\n", ls_net_perf)
            self.logger.info("因子报告生成完成: %s", self.name)
        except Exception:
            self.logger.exception("因子报告生成失败: %s", self.name)
            raise


if __name__ == "__main__":

    class my_factor(Factor):
        def caculate(self):
            MA, CLOSE = self.BIND("MA", "CLOSE")
            return MA(CLOSE, 20) / CLOSE - 1

    from config import db_path

    data = pd.read_parquet(db_path + "stocks_daily.parquet")
    data = data[data["date"].between(datetime(2023, 1, 1), datetime(2025, 12, 31))]
    data = data.set_index(["date", "symbol"]).sort_index()
    data["close"] = data["close"] * data["adj_factor"]
    ma_factor = my_factor("my_factor", data)
    transaction_cost_bps = 5
    ma_factor.create_factor_analysis_report(
        winsorize="3sigma",
        standardize=True,
        transaction_cost_bps=transaction_cost_bps,
    )
