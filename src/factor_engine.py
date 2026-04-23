from abc import ABC, abstractmethod
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
        # 1. 获取因子值
        factor = self.caculate()

        # 2. 去极值
        if winsorize == "3sigma":
            factor = factor.groupby(level="date").transform(
                lambda x: self.winsorize_3sigma(x, winsorize_param)
            )
        elif winsorize == "mad":
            factor = factor.groupby(level="date").transform(
                lambda x: self.winsorize_mad(x, winsorize_param)
            )

        # 3. 标准化
        if standardize:
            factor = factor.groupby(level="date").transform(self.standardize_zscore)

        # 4. 获取价格数据用于计算远期收益
        price = self.data["close"]

        # 5. 构建结果DataFrame（对齐索引）
        result = pd.DataFrame({"factor": factor, "close": price})

        # 6. 计算远期收益
        result["next_ret"] = (
            result["close"].groupby(level="symbol").pct_change(period).shift(-period)
        )

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

    def create_factor_analysis_report(
        self,
        quantiles: int = 5,
        period: int = 1,
        winsorize: str | None = None,
        winsorize_param: float = 3,
        standardize: bool = False,
    ) -> None:
        """生成完整的因子分析报告，包含 IC 时序、分组收益、多空组合等图表。

        Args:
            quantiles: 因子每日截面分组数（默认5）
            period: 远期收益计算周期天数（默认1）
            winsorize: 去极值方法，"3sigma" 或 "mad"，None 跳过
            winsorize_param: 去极值参数（默认3）
            standardize: 是否做 Z-score 标准化（默认 False）
        """
        result = self.get_clean_factor_and_forward_returns(
            quantiles, period, winsorize, winsorize_param, standardize
        )
        daily_group_ret = self.calculate_daily_group_ret(result)
        group_perf = self.calculate_all_group_performance(daily_group_ret, period)
        ls_ret, ls_perf = self.calculate_long_short(daily_group_ret, period)
        daily_ic = self.caculate_daily_ic(result)
        ic_stats = self.ic_statistics_analysis(daily_ic)
        ic_t = self.ic_mean_t_test(daily_ic)
        page = Page(page_title="因子分析报告", layout=Page.SimplePageLayout)
        page.add(self.plot_group_cumulative_return(daily_group_ret, ls_ret))
        page.add(self.plot_ic_time_series(daily_ic))
        page.add(self.plot_ic_cumulative(daily_ic))
        page.add(self.plot_group_annual_bar(group_perf))
        page.add(self.plot_ic_histogram(daily_ic))
        page.render("./output/因子分析报告.html")

        print("\n分组绩效指标\n", group_perf)
        print("\n多空组合指标\n", ls_perf)
        print("\nIC统计指标\n", ic_stats)
        print("\nIC T检验\n", ic_t)


if __name__ == "__main__":

    class ma(Factor):
        def caculate(self):
            return (
                self.data["turnover_rate"]
                .groupby(level="symbol")
                .transform(lambda x: -x.rolling(20).mean() / x)
            )

    from config import db_path

    data = pd.read_parquet(db_path + "stocks_daily.parquet")
    data = data[data["date"].between(datetime(2023, 1, 1), datetime(2023, 12, 31))]
    data = data.set_index(["date", "symbol"]).sort_index()
    data["close"] = data["close"] * data["adj_factor"]
    ma_factor = ma("ma", data)
    ma_factor.create_factor_analysis_report(winsorize="3sigma", standardize=True)
