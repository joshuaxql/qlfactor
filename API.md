# qlfactor API 文档

本文档按模块详细说明当前项目中的主要函数、方法、输入输出和注意事项。

## 1. src/config.py

### 1.1 模块作用

负责读取环境变量并初始化 Tushare Pro 客户端。

### 1.2 全局对象

1. token
- 类型：str | None
- 来源：环境变量 TUSHARE_TOKEN
- 作用：Tushare 鉴权 token。

2. db_path
- 类型：str
- 来源：环境变量 DB_NAME（默认 ./data/）
- 作用：数据文件存储目录。

3. pro
- 类型：tushare.pro client
- 生成条件：token 和 db_path 均存在。
- 作用：全局复用的 Tushare Pro API 客户端。

4. 异常
- 当 token 或 db_path 缺失时，抛出 ValueError。

## 2. src/download.py

## 2.1 类 Download

封装行情数据下载和合并流程。

### 2.1.1 __init__(self, pro, db_path: str) -> None

功能：
- 保存 Tushare 客户端和数据路径。

参数：
- pro：Tushare Pro 客户端。
- db_path：数据目录。

返回：
- 无。

### 2.1.2 calendar(self) -> None

功能：
- 拉取 SSE 开市日交易日历并保存为 calendar.parquet。

主要逻辑：
- 读取 1900-01-01 到 2050-01-01 的开市日。
- 字段 cal_date 转 datetime。
- 逆序后保存到 db_path/calendar.parquet。

返回：
- 无。

### 2.1.3 stocks_info(self) -> None

功能：
- 下载两市股票基础信息（L/D/P 状态）并合并。

主要逻辑：
- 分别请求 SSE、SZSE 的 L/D/P。
- 合并后将 ts_code 重命名为 symbol。
- list_date 转 datetime。
- 保存到 db_path/stocks_info.parquet。

返回：
- 无。

### 2.1.4 stocks_daily(self) -> None

功能：
- 按股票下载历史日线、复权因子、daily_basic，并合并股票曾用名。

主要逻辑：
- 从 stocks_info.parquet 获取股票列表与上市日期。
- 按交易日历切片，每 5000 个交易日分块请求。
- 合并 daily、adj_factor、daily_basic。
- 请求 namechange 并按时间 merge_asof 映射历史简称。
- 统一字段名并输出每只股票文件到 db_path/stocks_daily/{symbol}.parquet。
- 单只股票失败时重试最多 3 次。

输入依赖：
- calendar.parquet
- stocks_info.parquet

返回：
- 无。

注意：
- 该方法网络与接口调用量较大，耗时长。

### 2.1.5 merge(self) -> None

功能：
- 读取 stocks_daily 目录下所有个股 parquet，并合并为总表。

主要逻辑：
- 并行读取 parquet（ProcessPoolExecutor）。
- 合并并按 date/symbol 排序。
- 保存为 db_path/stocks_daily.parquet。

返回：
- 无。

## 3. src/factor_engine.py

## 3.1 抽象类 Factor

职责：
- 定义因子计算接口。
- 提供通用的因子清洗、收益分析、IC 分析、换手率分析、交易成本分析、图表输出和报告生成。

数据约定：
- self.data 为 MultiIndex(date, symbol) 的 DataFrame。
- 至少包含 close 字段；示例因子使用 turnover_rate。

### 3.1.1 __init__(self, name: str, data: pd.DataFrame) -> None

功能：
- 初始化因子名称和输入数据。

参数：
- name：因子名。
- data：日线数据，索引为 (date, symbol)。

### 3.1.2 caculate(self) -> pd.Series

功能：
- 抽象方法，由子类实现具体因子值计算。

返回：
- 因子值 Series，索引为 (date, symbol)。

### 3.1.3 winsorize_3sigma(self, factor: pd.Series, n: int = 3) -> pd.Series

功能：
- 3sigma 去极值。

规则：
- 截断到 [mean - n*std, mean + n*std]。

### 3.1.4 winsorize_mad(self, factor: pd.Series, k: float = 3) -> pd.Series

功能：
- MAD 去极值。

规则：
- 截断到 [median - k*MAD, median + k*MAD]。

### 3.1.5 standardize_zscore(self, factor: pd.Series) -> pd.Series

功能：
- Z-Score 标准化。

规则：
- (x - mean) / std。
- 当 std=0 或缺失时返回全 0 序列。

### 3.1.6 _prepare_factor_values(self, winsorize: str | None = None, winsorize_param: float = 3, standardize: bool = False) -> pd.Series

功能：
- 对子类计算出的原始因子值执行清洗预处理。

流程：
1. 调用 caculate。
2. 按 date 截面去极值（可选）。
3. 按 date 截面标准化（可选）。

返回：
- 预处理后的因子 Series。

### 3.1.7 get_clean_factor_and_forward_returns(self, quantiles: int = 5, period: int = 1, winsorize: str | None = None, winsorize_param: float = 3, standardize: bool = False) -> pd.DataFrame

功能：
- 生成因子分析主数据集。

流程：
1. 预处理因子值。
2. 构造 factor/close 对齐表。
3. 计算远期收益 next_ret。
- 先按 symbol 计算 pct_change(period)
- 再按 symbol 做 shift(-period)
4. 删除缺失。
5. 按 date 截面做分位分组。

返回：
- DataFrame，核心列：factor、close、next_ret、factor_quantile。

### 3.1.8 caculate_daily_ic(self, result: pd.DataFrame) -> pd.Series

功能：
- 计算日度 Rank IC（Spearman）。

规则：
- 当天因子或收益唯一值不足 2 个时返回 NaN。

### 3.1.9 ic_statistics_analysis(self, daily_ic: pd.Series) -> pd.Series

功能：
- 汇总 IC 统计：均值、标准差、IR、年化 IR。

### 3.1.10 ic_mean_t_test(self, daily_ic: pd.Series) -> pd.Series

功能：
- 对 IC 均值做单样本 t 检验（原假设均值=0）。

返回：
- T 统计量、P 值、显著性文本。

### 3.1.11 plot_ic_time_series(self, daily_ic: pd.Series) -> Line

功能：
- 绘制每日 IC 折线图与均值线。

### 3.1.12 plot_ic_cumulative(self, daily_ic: pd.Series) -> Line

功能：
- 绘制 IC 累加曲线。

### 3.1.13 calculate_daily_group_ret(self, result: pd.DataFrame) -> pd.DataFrame

功能：
- 计算各分组的日度等权平均收益。

返回：
- 行索引 date、列为组1~组N。

### 3.1.14 calculate_group_turnover(self, result: pd.DataFrame) -> pd.DataFrame

功能：
- 计算分组持仓换手率。

定义：
- 换手率 = 1 - 前后两日持仓重合比例。

### 3.1.15 calculate_group_turnover_stats(self, group_turnover: pd.DataFrame) -> pd.DataFrame

功能：
- 汇总分组换手率统计。

输出列：
- 平均换手率
- 换手率标准差
- P25
- P50
- P75

### 3.1.16 calculate_factor_turnover_rate(self, factor: pd.Series) -> pd.Series

功能：
- 计算因子整体换手率。

定义：
- 因子换手率 = 1 - 截面秩自相关。

实现细节：
- 先转宽表并做截面百分位秩。
- 与前一日秩做 Spearman 相关。
- 常量截面自动记为 NaN。

### 3.1.17 factor_turnover_stats(self, factor_turnover: pd.Series) -> pd.Series

功能：
- 汇总因子换手率统计。

输出：
- 平均换手率、换手率标准差、P25/P50/P75。

### 3.1.18 calculate_turnover_return_correlation(self, group_turnover: pd.DataFrame, group_ret: pd.DataFrame) -> pd.Series

功能：
- 计算各分组换手率与收益的相关系数。

返回：
- Series，索引为组名。

### 3.1.19 calculate_net_returns_with_cost(self, group_ret: pd.DataFrame, group_turnover: pd.DataFrame, cost_bps: float = 10) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]

功能：
- 按换手率扣交易成本，输出净收益。

参数：
- cost_bps：单边成本，单位 bps。

返回：
1. 分组净收益 DataFrame
2. 多空毛收益 Series
3. 多空净收益 Series
4. 多空净绩效 Series

公式：
- 成本率 = cost_bps / 10000
- 分组净收益 = 分组毛收益 - 分组换手率 × 成本率
- 多空净收益 = 多空毛收益 - (高组换手率 + 低组换手率) × 成本率

### 3.1.20 performance_analysis(self, ret_series: pd.Series, period: int = 1) -> pd.Series

功能：
- 单序列绩效统计。

输出：
- 年化收益
- 年化波动率
- 夏普比率
- 总累计收益

### 3.1.21 calculate_all_group_performance(self, group_ret: pd.DataFrame, period: int = 1) -> pd.DataFrame

功能：
- 批量计算各分组绩效。

### 3.1.22 calculate_long_short(self, group_ret: pd.DataFrame, period: int = 1) -> tuple[pd.Series, pd.Series]

功能：
- 计算多空组合（最高组 - 最低组）收益与绩效。

### 3.1.23 plot_group_annual_bar(self, group_perf_df: pd.DataFrame) -> Bar

功能：
- 分组年化收益柱状图。

### 3.1.24 plot_ic_histogram(self, daily_ic: pd.Series) -> Bar

功能：
- IC 分布直方图。

### 3.1.25 plot_group_cumulative_return(self, group_ret: pd.DataFrame, long_short_ret: pd.Series) -> Line

功能：
- 分组累计收益曲线 + 多空累计净值曲线。

### 3.1.26 plot_group_turnover(self, group_turnover: pd.DataFrame) -> Line

功能：
- 分组换手率时序图。

### 3.1.27 plot_group_turnover_bar(self, turnover_stats: pd.DataFrame) -> Bar

功能：
- 分组平均换手率柱状图。

### 3.1.28 plot_factor_turnover(self, factor_turnover: pd.Series) -> Line

功能：
- 因子整体换手率时序图。

### 3.1.29 plot_turnover_return_correlation(self, corr_series: pd.Series) -> Bar

功能：
- 分组换手率与收益相关系数柱状图。

### 3.1.30 plot_long_short_net_vs_gross(self, long_short_gross: pd.Series, long_short_net: pd.Series, cost_bps: float) -> Line

功能：
- 多空毛收益累计、净收益累计、累计成本拖累三条曲线对比。

### 3.1.31 create_factor_analysis_report(self, quantiles: int = 5, period: int = 1, winsorize: str | None = None, winsorize_param: float = 3, standardize: bool = False, transaction_cost_bps: float = 10) -> None

功能：
- 一键生成完整分析流程和 HTML 报告。

核心步骤：
1. 构建因子数据集。
2. 计算分组收益、多空收益、IC、换手率、净收益。
3. 计算分组收益、多空收益、IC、换手率与净收益。
4. 生成可视化页面并写入 output/因子分析报告.html。
5. 在控制台输出关键统计表。

关键参数：
- transaction_cost_bps：单边交易成本（bps）。

## 4. 主程序示例（factor_engine.py 中）

示例子类 ma 的因子定义为：
- 20 日平均换手率 / 当日换手率。

示例流程：
1. 读取 stocks_daily.parquet。
2. 筛选 2023 年。
3. 设置索引并复权 close。
4. 调用 create_factor_analysis_report。

## 5. 术语与口径说明

1. 单边成本
- 指一次买入或卖出单独收取的成本。

2. bps
- 1 bps = 0.01%。
- 10 bps = 0.10%。

3. 因子换手率
- 用截面秩的日度自相关衡量因子稳定性，数值越高通常代表持仓变化更快。

4. 分组换手率
- 基于分组成分股集合变化计算，反映组合调仓强度。

## 6. 调用示例片段

以下示例均为最小可运行片段，默认在项目根目录执行。

### 6.1 config 模块

```python
from src.config import pro, db_path

print(type(pro))
print(db_path)
```

### 6.2 Download.__init__

```python
from src.config import pro, db_path
from src.download import Download

dl = Download(pro, db_path)
print(dl.db_path)
```

### 6.3 Download.calendar

```python
from src.config import pro, db_path
from src.download import Download

dl = Download(pro, db_path)
dl.calendar()
print("calendar.parquet done")
```

### 6.4 Download.stocks_info

```python
from src.config import pro, db_path
from src.download import Download

dl = Download(pro, db_path)
dl.stocks_info()
print("stocks_info.parquet done")
```

### 6.5 Download.stocks_daily

```python
from src.config import pro, db_path
from src.download import Download

dl = Download(pro, db_path)
dl.stocks_daily()
print("stocks_daily/*.parquet done")
```

### 6.6 Download.merge

```python
from src.config import pro, db_path
from src.download import Download

dl = Download(pro, db_path)
dl.merge()
print("stocks_daily.parquet done")
```

### 6.7 Factor 子类最小实现（对应 caculate）

```python
import pandas as pd
from src.factor_engine import Factor

class DemoFactor(Factor):
	def caculate(self):
		# 20日均换手 / 当日换手
		return (
			self.data["turnover_rate"]
			.groupby(level="symbol")
			.transform(lambda x: x.rolling(20).mean() / x)
		)

data = pd.read_parquet("./data/stocks_daily.parquet")
data = data.set_index(["date", "symbol"]).sort_index()
data["close"] = data["close"] * data["adj_factor"]
f = DemoFactor("demo", data)
print(f.caculate().dropna().head())
```

### 6.8 winsorize_3sigma / winsorize_mad / standardize_zscore

```python
factor_raw = f.caculate().dropna()

sample = factor_raw.groupby(level="date").head(200)
w1 = f.winsorize_3sigma(sample, n=3)
w2 = f.winsorize_mad(sample, k=3)
z = f.standardize_zscore(sample)

print(w1.head())
print(w2.head())
print(z.head())
```

### 6.9 _prepare_factor_values

```python
prepared = f._prepare_factor_values(
	winsorize="3sigma",
	winsorize_param=3,
	standardize=True,
)
print(prepared.dropna().head())
```

### 6.10 get_clean_factor_and_forward_returns

```python
result = f.get_clean_factor_and_forward_returns(
	quantiles=5,
	period=1,
	winsorize="3sigma",
	winsorize_param=3,
	standardize=True,
)
print(result.head())
```

### 6.11 caculate_daily_ic / ic_statistics_analysis / ic_mean_t_test

```python
daily_ic = f.caculate_daily_ic(result)
ic_stats = f.ic_statistics_analysis(daily_ic)
ic_t = f.ic_mean_t_test(daily_ic)

print(daily_ic.dropna().head())
print(ic_stats)
print(ic_t)
```

### 6.12 calculate_daily_group_ret / calculate_long_short / performance_analysis / calculate_all_group_performance

```python
group_ret = f.calculate_daily_group_ret(result)
ls_ret, ls_perf = f.calculate_long_short(group_ret, period=1)
group_perf = f.calculate_all_group_performance(group_ret, period=1)
ls_perf_check = f.performance_analysis(ls_ret, period=1)

print(group_ret.head())
print(ls_perf)
print(group_perf.head())
print(ls_perf_check)
```

### 6.13 calculate_group_turnover / calculate_group_turnover_stats

```python
group_turnover = f.calculate_group_turnover(result)
group_turnover_stats = f.calculate_group_turnover_stats(group_turnover)

print(group_turnover.head())
print(group_turnover_stats)
```

### 6.14 calculate_factor_turnover_rate / factor_turnover_stats

```python
factor_values = f._prepare_factor_values(winsorize="3sigma", standardize=True).dropna()
factor_turnover = f.calculate_factor_turnover_rate(factor_values)
factor_turnover_stats = f.factor_turnover_stats(factor_turnover)

print(factor_turnover.dropna().head())
print(factor_turnover_stats)
```

### 6.15 calculate_turnover_return_correlation

```python
corr_series = f.calculate_turnover_return_correlation(group_turnover, group_ret)
print(corr_series)
```

### 6.16 calculate_net_returns_with_cost

```python
group_net_ret, ls_gross_ret, ls_net_ret, ls_net_perf = f.calculate_net_returns_with_cost(
	group_ret,
	group_turnover,
	cost_bps=10,
)

print(group_net_ret.head())
print(ls_gross_ret.head())
print(ls_net_ret.head())
print(ls_net_perf)
```

### 6.17 图表函数最小调用

```python
chart1 = f.plot_ic_time_series(daily_ic)
chart2 = f.plot_ic_cumulative(daily_ic)
chart3 = f.plot_ic_histogram(daily_ic)
chart4 = f.plot_group_annual_bar(group_perf)
chart5 = f.plot_group_cumulative_return(group_ret, ls_ret)
chart6 = f.plot_group_turnover(group_turnover)
chart7 = f.plot_group_turnover_bar(group_turnover_stats)
chart8 = f.plot_factor_turnover(factor_turnover)
chart9 = f.plot_turnover_return_correlation(corr_series)
chart10 = f.plot_long_short_net_vs_gross(ls_gross_ret, ls_net_ret, cost_bps=10)

for c in [chart1, chart2, chart3, chart4, chart5, chart6, chart7, chart8, chart9, chart10]:
	print(type(c))
```

### 6.18 create_factor_analysis_report

```python
f.create_factor_analysis_report(
	quantiles=5,
	period=1,
	winsorize="3sigma",
	winsorize_param=3,
	standardize=True,
	transaction_cost_bps=10,
)

print("report done: ./output/因子分析报告.html")
```
