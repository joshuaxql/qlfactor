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
        self.name = name
        self.data = data

    @abstractmethod
    def caculate(self): ...

    def get_clean_factor_and_forward_returns(
        self,
        quantiles: int = 5,
        period: int = 1,
    ) -> pd.DataFrame:
        """清洗因子数据并计算多周期远期收益，生成因子分析标准数据集。

        Args:
            quantiles: 因子每日截面分组数（按因子值分组，如分5组）
            periods: 换仓周期列表，规定远期收益计算间隔（如[1, 5, 20]）
            price_factor: 价格依赖因子的键名，用于计算远期收益

        Returns:
            干净的长格式 DataFrame，列包含：
            date, symbol, factor, 各周期远期收益 (1D, 5D, 20D...), factor_quantile
        """
        # 1. 获取因子值
        factor = self.caculate()

        # 2. 获取价格数据用于计算远期收益
        price = self.data["close"]

        # 3. 构建结果DataFrame（对齐索引）
        result = pd.DataFrame({"factor": factor, "close": price})

        # 4. 计算远期收益
        result["next_ret"] = (
            result["close"].groupby(level="symbol").pct_change(period).shift(-period)
        )

        # 5. 清洗：去除NaN
        result = result.dropna()

        # 6. 计算因子分位数（每日截面）
        result["factor_quantile"] = result.groupby(level="date")["factor"].transform(
            lambda x: pd.qcut(x, q=quantiles, labels=False, duplicates="drop") + 1
        )

        return result

    def caculate_daily_ic(self, result: pd.DataFrame):
        daily_ic = result.groupby(level="date").apply(
            lambda x: x["factor"].corr(x["next_ret"], method="spearman")
        )
        daily_ic.name = "IC"

        return daily_ic

    def ic_statistics_analysis(self, daily_ic):
        """
        IC 统计指标分析
        :param daily_ic: 每日IC序列
        :return: IC统计结果
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
            }
        )
        return ic_analysis

    def ic_mean_t_test(self, daily_ic):
        """
        对每日IC序列进行单样本T检验
        检验：IC均值是否显著不等于0
        返回：T统计量、p值、显著性结论
        """
        # 剔除空值
        ic = daily_ic.dropna()
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

    def plot_ic_time_series(self, daily_ic):
        """
        绘制因子每日IC时序图（无数据点、无标签，纯净版）
        :param daily_ic: 每日IC序列
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

    def plot_ic_cumulative(self, daily_ic):
        """
        绘制 IC 累加净值曲线（清爽版）
        IC累加 = 每日IC依次相加，用于观察因子持续有效性
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

    def calculate_daily_group_ret(self, result: pd.DataFrame):
        """
        按【日期+因子分组】计算每日等权平均收益
        :param df: 预处理后的DataFrame
        :return: 日期为行、分组为列的每日收益表
        """
        group_daily = (
            result.groupby(["date", "factor_quantile"])["next_ret"].mean().reset_index()
        )
        group_ret = group_daily.pivot(
            index="date", columns="factor_quantile", values="next_ret"
        )
        group_ret.columns = [f"组{i}" for i in group_ret.columns]
        return group_ret

    def performance_analysis(self, ret_series):
        """
        计算单组收益的核心量化指标
        :param ret_series: 日收益序列
        :return: 年化收益、波动率、夏普、总收益
        """
        ann_ret = ret_series.mean() * 252  # 年化收益（A股252交易日）
        ann_vol = ret_series.std() * np.sqrt(252)  # 年化波动率
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

    def calculate_all_group_performance(self, group_ret):
        """
        计算所有分组的绩效指标汇总表
        :param daily_ret_df: 每日分组收益表
        :return: 分组绩效汇总DataFrame
        """
        return group_ret.apply(self.performance_analysis).T

    def calculate_long_short(self, group_ret):
        """
        计算多空组合收益（最高因子组 - 最低因子组）
        :param group_ret: 每日分组收益表
        :return: 多空日收益、多空绩效指标
        """
        long_short_ret = group_ret["组5"] - group_ret["组1"]
        long_short_perf = self.performance_analysis(long_short_ret)
        long_short_perf.name = "多空组合(组5-组1)"
        return long_short_ret, long_short_perf

    def plot_group_annual_bar(self, group_perf_df):
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

    def plot_ic_histogram(self, daily_ic):
        hist, bins = np.histogram(daily_ic, bins=30)
        bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
        bar.add_xaxis(
            [round((bins[i] + bins[i + 1]) / 2, 3) for i in range(len(bins) - 1)]
        )
        bar.add_yaxis("IC频次", hist.tolist(), label_opts=opts.LabelOpts(is_show=False))
        bar.set_global_opts(
            title_opts=opts.TitleOpts(title="IC分布直方图"),
            xaxis_opts=opts.AxisOpts(name="IC值"),
            yaxis_opts=opts.AxisOpts(name="频次"),
        )
        return bar

    def plot_group_cumulative_return(self, group_ret, long_short_ret):
        """
        Pyecharts 绘制因子分组累计收益+多空组合净值曲线（交互式）
        :param group_ret: 分组累计收益DataFrame
        :param long_short_ret: 多空日收益序列
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

        line.add_yaxis(
            series_name="多空组合(组5-组1)",
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
    ):
        result = self.get_clean_factor_and_forward_returns(quantiles, period)
        daily_group_ret = self.calculate_daily_group_ret(result)
        group_perf = self.calculate_all_group_performance(daily_group_ret)
        ls_ret, ls_perf = self.calculate_long_short(daily_group_ret)
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
    from config import db_path
    import glob
    import os
    from tqdm import tqdm

    stocks = [
        os.path.basename(f).replace(".parquet", "")
        for f in glob.glob(db_path + "daily/*.parquet")
    ]
    data = pd.DataFrame()
    for stock in tqdm(stocks[:50]):
        temp = pd.read_parquet(db_path + f"daily/{stock}.parquet")
        temp = temp[temp["date"].between(datetime(2024, 1, 1), datetime(2026, 1, 1))]
        data = pd.concat([data, temp])
    data["close"] = data["close"] * data["adj_factor"]
    data = data[["symbol", "date", "close", "turnover_rate_f"]]
    data = data.set_index(["date", "symbol"])

    class ma(Factor):
        def caculate(self):
            return (
                self.data["turnover_rate_f"]
                .groupby(level="symbol")
                .transform(lambda x: -x.rolling(20).mean())
            )

    ma_factor = ma("ma", data)
    ma_factor.create_factor_analysis_report()
