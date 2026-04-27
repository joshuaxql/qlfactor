# qlfactor API 文档

本文档按模块详细说明当前项目中的主要函数、方法、输入输出和注意事项。

## 1. src/config.py

### 1.1 模块作用

负责配置加载与日志初始化，且在 import 时无副作用。

### 1.2 数据结构与函数

1. `Config`
- 字段：`token: str`、`db_path: pathlib.Path`。
- 行为：`__post_init__` 会规范化路径（绝对化、展开 `~`）。
- 惰性属性：`pro` 首次访问时才创建 `tushare.pro_api(token)` 客户端。

2. `load_config(env_file=None, create_db_dir=True) -> Config`
- 功能：读取 `.env`（或显式路径）并返回 `Config`。
- 必填项：`TUSHARE_TOKEN` 与 `DB_NAME`。
- 异常：任一缺失会抛 `ValueError`。
- 目录：`create_db_dir=True` 时自动创建 `db_path`。

3. `setup_logging(level=logging.INFO, log_file="debug.log", fmt=..., force=False) -> None`
- 功能：初始化根日志，默认同时输出到终端和 `debug.log`。
- 特性：幂等；`force=True` 可重建 handler。

4. 设计约束
- 不再提供 import 即可使用的全局 `token/db_path/pro` 变量。
- 推荐调用链：`setup_logging()` → `cfg = load_config()` → `cfg.pro`。

## 2. src/download.py

## 2.1 类 Download

封装行情数据下载与合并流程。

### 2.1.1 __init__(self, pro: Any, db_path: Path | str) -> None

功能：保存 Tushare 客户端与数据目录，并自动创建目录。

### 2.1.2 from_config(cls, config: Config) -> Download

功能：从 `load_config()` 返回的 `Config` 直接构造下载器。

### 2.1.3 calendar(self) -> None

功能：下载 SSE 开市日交易日历，写入 `calendar.parquet`。

### 2.1.4 stocks_info(self) -> None

功能：下载 SSE/SZSE 全市场股票基础信息（L/D/P），写入 `stocks_info.parquet`。

### 2.1.5 industry(self, level: str = "L1", src: str = "SW2021") -> None

功能：下载申万行业成分变更表，写入 `industry.parquet`。

### 2.1.6 stocks_daily(self, basic: bool = True) -> None

功能：逐股下载日线、复权因子、（可选）daily_basic，并合并曾用名。

主要逻辑：
1. 依赖 `calendar.parquet` 与 `stocks_info.parquet`。
2. 按 5000 交易日分块请求 daily/adj_factor/daily_basic。
3. 每只股票最多重试 3 次；失败仅 warning 后继续。
4. 输出到 `stocks_daily/{symbol}.parquet`。

### 2.1.7 merge(self) -> None

功能：并行读取 `stocks_daily/*.parquet` 并合并为 `stocks_daily.parquet`。

### 2.1.8 产物清单

- `calendar.parquet`
- `stocks_info.parquet`
- `industry.parquet`
- `stocks_daily/{symbol}.parquet`
- `stocks_daily.parquet`

## 3. src/factor_engine.py

## 3.1 抽象类 Factor

职责：
- 定义因子计算接口。
- 提供通用的因子清洗、收益分析、IC 分析、换手率分析、交易成本分析、图表输出和报告生成。
- 提供公式化因子 DSL（如 `MA(CLOSE, 20)`）及通达信风格算子。
- 在公式执行、因子预处理、报告生成阶段输出日志，便于排错。

数据约定：
- `self.data` 为 MultiIndex `(date, symbol)` 的 DataFrame，且应先 `sort_index()`。
- 至少包含 `close`；当 `adjust=True` 时还必须包含 `adj_factor`。
- `adjust=True` 会在因子计算前先复权 `open/high/low/close/pre_close`，再计算因子。

### 3.1.1 __init__(self, name: str, data: pd.DataFrame) -> None

功能：
- 初始化因子名称和输入数据。

参数：
- name：因子名。
- data：日线数据，索引为 (date, symbol)。

### 3.1.2 calculate(self) -> pd.Series

功能：
- 抽象方法，由子类实现具体因子值计算。

返回：
- 因子值 Series，索引为 (date, symbol)。

### 3.1.2A 字段别名（大写属性）

可直接在公式里使用：

- OPEN, HIGH, LOW, CLOSE
- PRE_CLOSE, CHANGE, PCT_CHG
- VOLUME, AMOUNT, ADJ_FACTOR
- TURNOVER_RATE, TURNOVER_RATE_F, VOLUME_RATIO
- PE, PE_TTM, PB, PS, PS_TTM
- DV_RATIO, DV_TTM
- TOTAL_SHARE, FLOAT_SHARE, FREE_SHARE
- TOTAL_MV, CIRC_MV

### 3.1.2B 公式入口：FORMULA / BIND

1. `FORMULA(self, expr: str, **extra) -> pd.Series`
- 功能：在安全命名空间执行表达式字符串。
- 示例：`self.FORMULA("MA(CLOSE, 20) / CLOSE - 1")`
- 返回：Series（标量会自动扩展为与 `self.data.index` 对齐的 Series）。
- 异常：表达式执行失败时会记录 traceback 并抛出原异常。

2. `BIND(self, *names: str)`
- 功能：绑定公式名称，减少重复写 `self.`。
- 示例：`MA, CLOSE = self.BIND("MA", "CLOSE")`

### 3.1.2C 公式函数详解

所有公式函数都以方法形式挂在 `Factor` 实例上，并通过 `_formula_namespace`
注入到 `FORMULA(...)` 与 `BIND(...)` 的执行环境中。下面按类别给出每个函数的
**签名、口径、参数、返回、注意事项与最小示例**。

通用约定：

- 形参 `x / a / b` 既可传字段别名字符串（自动取 `self.data[小写]`），也可传
  任意 `pd.Series`（与 `self.data.index` 对齐）。`MAX/MIN/IF/COUNT/EVERY/EXIST/CROSS`
  与 `CORRELATION/COVARIANCE/SIGN` 还支持标量。
- `n / m` 必须是 `int`；返回值始终是 MultiIndex `(date, symbol)` 的 Series
  （`EVERY/EXIST/CROSS` 返回布尔 Series）。
- 时序滚动一律按 `symbol` 分组；截面操作（`RANK`）按 `date` 分组。
- `min_periods=None` 表示与窗口长度 `n` 相等（要满窗才输出）；传整数则放宽，
  早期窗口不足时也会输出。
- 条件类参数（`cond`）会被 `_resolve_value` 处理为布尔，`NaN` 当作 `False`。

#### 基础时序 / 截面算子

##### MA(x, n, min_periods=None)

简单移动平均，按 `symbol` 分组的窗口长度 `n` 滚动均值。

- 参数：
  - `x`：字段别名或 Series。
  - `n`：窗口长度。
  - `min_periods`：默认 `n`（满窗才输出），可放宽。
- 返回：与 `x` 同索引的 Series。
- 示例：
  ```python
  self.FORMULA("MA(CLOSE, 20)")              # 20 日均价
  self.FORMULA("CLOSE / MA(CLOSE, 20) - 1")  # 相对均线偏离
  ```

##### EMA(x, n, adjust=True, min_periods=1)

指数移动平均（基于 `pandas.ewm(span=n)`），按 `symbol` 分组。

- 参数：
  - `n`：等效 SMA 窗口，对应 `alpha = 2/(n+1)`。
  - `adjust`：与 `pandas.ewm` 含义相同；想得到通达信 EMA 的递归口径请传
    `adjust=False`。
  - `min_periods`：默认 `1`，第一个观测即可输出。
- 示例：
  ```python
  self.FORMULA("EMA(CLOSE, 12) - EMA(CLOSE, 26)")  # MACD diff
  ```

##### RANK(x, pct=True, ascending=True, method="average")

**截面排序**：按 `date` 对当日所有股票排名。

- 参数：
  - `pct=True` 输出 0-1 百分比秩，`False` 输出 1..N 名次。
  - `ascending=True` 升序（值越小秩越低）。
  - `method`：`"average" | "min" | "max" | "first" | "dense"`，对应 `pandas.rank`。
- 与 `TSRANK` 区分：`RANK` 是**截面**（同日股票之间），`TSRANK` 是**时序**
  （同股票最近 n 天之间）。
- 示例：
  ```python
  self.FORMULA("RANK(VOLUME)")              # 当日成交量分位
  self.FORMULA("RANK(MA(CLOSE, 20))")       # 20 日均价的截面分位
  ```

##### REF(x, n=1)

滞后 n 期，等价于 `groupby(symbol).shift(n)`。前 n 期为 NaN。

- 示例：
  ```python
  self.FORMULA("CLOSE / REF(CLOSE, 1) - 1")  # 隐式日收益率
  ```

##### DELTA(x, n=1)

按 `symbol` 的差分 `x_t - x_{t-n}`。前 n 期为 NaN。

- 示例：
  ```python
  self.FORMULA("DELTA(CLOSE, 5)")           # 5 日价差
  ```

##### STD(x, n, min_periods=None)

按 `symbol` 的滚动标准差（默认 ddof=1，与 pandas 一致）。

- 示例：
  ```python
  self.FORMULA("STD(PCT_CHG, 20)")          # 20 日波动率
  ```

##### SUM(x, n, min_periods=None)

按 `symbol` 的滚动求和。

- 示例：
  ```python
  self.FORMULA("SUM(VOLUME, 5)")            # 5 日累计成交量
  ```

##### CORRELATION(x, y, n, min_periods=None)

按 `symbol` 的滚动相关系数，等价于 `rolling(...).corr(...)`。

- 参数：
  - `x / y`：字段别名、Series 或标量（会先对齐到 `self.data.index`）。
  - `n`：窗口长度。
  - `min_periods`：默认 `n`。
- 返回：与输入索引对齐的相关系数 Series。
- 示例：
  ```python
  self.FORMULA("CORRELATION(CLOSE, VOLUME, 20)")  # 价量相关
  ```

##### COVARIANCE(x, y, n, min_periods=None)

按 `symbol` 的滚动协方差，等价于 `rolling(...).cov(...)`。

- 参数与 `CORRELATION` 一致。
- 示例：
  ```python
  self.FORMULA("COVARIANCE(PCT_CHG, VOLUME, 20)")  # 收益与成交量协方差
  ```

#### 通达信风格函数

##### ABS(x)

逐元素绝对值。

- 示例：`self.FORMULA("ABS(DELTA(CLOSE, 1))")`

##### SIGN(x)

逐元素符号函数，返回 `-1 / 0 / 1`（NaN 保持 NaN）。

- 示例：`self.FORMULA("SIGN(DELTA(CLOSE, 1))")`

##### MAX(a, b)

**逐元素较大值**（不是滚动窗口最大值——那是 `HHV`）。`a/b` 都经 `_resolve_value`
解析，因此可以混合 Series 与标量。

- 示例：
  ```python
  self.FORMULA("MAX(CLOSE - REF(CLOSE, 1), 0)")  # 上行幅度（与 0 取大）
  ```

##### MIN(a, b)

逐元素较小值，语义与 `MAX` 镜像。

##### IF(cond, a, b)

三元：`np.where(cond, a, b)`。`cond` 的 NaN 当作 False；`a/b` 自动对齐索引。

- 示例：
  ```python
  self.FORMULA("IF(CLOSE > REF(CLOSE, 1), 1, -1)")  # 涨跌方向哑变量
  ```

##### COUNT(cond, n, min_periods=None)

`cond` 在最近 n 期里为 True 的**次数**（按 symbol 滚动）。

- 示例：
  ```python
  self.FORMULA("COUNT(CLOSE > REF(CLOSE, 1), 5)")  # 最近 5 日上涨天数
  ```

##### EVERY(cond, n, min_periods=None)

最近 n 期是否**全部**为 True，返回布尔 Series。窗口不足时按 False 处理。

- 示例：
  ```python
  self.FORMULA("EVERY(CLOSE > REF(CLOSE, 1), 3)")  # 连涨 3 天
  ```

##### EXIST(cond, n, min_periods=None)

最近 n 期内**至少一次**为 True。

- 示例：
  ```python
  self.FORMULA("EXIST(VOLUME > 1.5 * MA(VOLUME, 20), 5)")  # 5 日内出现过放量
  ```

##### HHV(x, n, min_periods=None)

最近 n 期最**高**值（按 symbol 滚动）。

- 示例：
  ```python
  # 当前价在 20 日箱体内的相对位置（0=底，1=顶）
  self.FORMULA("(CLOSE - LLV(CLOSE, 20)) / (HHV(CLOSE, 20) - LLV(CLOSE, 20))")
  ```

##### LLV(x, n, min_periods=None)

最近 n 期最**低**值。

##### TSRANK(x, n, pct=True, min_periods=None)

**时序排名**：当前值在最近 n 期窗口内的排名。

- `pct=True` 输出 0..1，`False` 输出 1..n。
- 与 `RANK`（截面）形成对偶；常用于动量类因子。
- 示例：
  ```python
  self.FORMULA("TSRANK(CLOSE, 60)")  # 当前价在过去 60 日中的相对位置
  ```

##### TS_ARGMAX(x, n, min_periods=None)

**时序最大值位置**：当前滚动窗口内最大值所在的 **1-based** 位置（`1=最早`，`n=最新`）。

- 若窗口内最大值出现多次，取最早出现的位置（与 `numpy.nanargmax` 一致）。
- 示例：
  ```python
  self.FORMULA("TS_ARGMAX(CLOSE, 20)")  # 20 日最高价距窗口起点的位置
  ```

##### SMA(x, n, m=1, min_periods=1)

**通达信 SMA**：递归权重平均 `Y = (m*X + (n-m)*Y_prev) / n`，等价于
`ewm(alpha=m/n, adjust=False)`。

- 约束：`n > 0` 且 `0 < m <= n`，否则抛 `ValueError`。
- 注意：此处 `SMA` **不是**简单移动平均（pandas 简单 MA 见 `MA`），它带指数衰减。
- 示例：
  ```python
  self.FORMULA("SMA(CLOSE, 12, 2)")  # KDJ 中的常见平滑
  ```

##### CROSS(a, b)

上穿信号：本期 `a > b` 且上期 `a <= b`，返回布尔 Series。

- 示例：
  ```python
  self.FORMULA("CROSS(MA(CLOSE, 5), MA(CLOSE, 20))")  # 5/20 金叉
  ```

#### 数学函数

##### LOG(x) / EXP(x) / SQRT(x)

直接绑定的 `np.log` / `np.exp` / `np.sqrt`，逐元素作用在 Series 上。`LOG` 与
`SQRT` 不会自动处理非正数——零或负数会得到 `-inf` / `nan`，需要时请自行
`x.where(x > 0)`。

- 示例：
  ```python
  self.FORMULA("LOG(TOTAL_MV)")             # 对数市值（市值中性化常用）
  self.FORMULA("SQRT(STD(PCT_CHG, 20))")    # 波动率开方
  ```

#### 速查表

| 函数                        | 类别       | 一句话                                |
| --------------------------- | ---------- | ------------------------------------- |
| `MA` / `EMA` / `SMA`        | 时序均值   | 简单 / 指数 / 通达信 SMA（指数加权）  |
| `STD` / `SUM`               | 时序聚合   | 滚动标准差 / 求和                     |
| `CORRELATION` / `COVARIANCE` | 双序列聚合 | 两序列滚动相关 / 协方差              |
| `REF` / `DELTA`             | 时序错位   | 滞后 / 差分                           |
| `HHV` / `LLV`               | 时序极值   | n 期最高 / 最低                       |
| `TSRANK` / `TS_ARGMAX`      | 时序排名   | 当前值排名 / 窗口最大值位置           |
| `RANK`                      | 截面排名   | 当日全市场分位                        |
| `MAX` / `MIN`               | 逐元素极值 | **不是**窗口极值（区别于 `HHV/LLV`）  |
| `ABS` / `SIGN` / `IF`       | 逐元素     | 绝对值 / 符号 / 三元                  |
| `COUNT` / `EVERY` / `EXIST` | 条件聚合   | n 期内 True 的 次数 / 全是 / 至少一次 |
| `CROSS`                     | 信号       | 上穿（同 `a>b` 且上期 `a<=b`）        |
| `LOG` / `EXP` / `SQRT`      | 数学       | numpy 同名函数，逐元素                |

#### 易混点提示

- `MAX(a, b)` vs `HHV(x, n)`：前者是两序列的逐元素较大值，后者是单序列的窗口最大。
  写动量类因子常要的是 `HHV`。
- `MA(x, n)` vs `SMA(x, n, m)`：`MA` 是简单等权均值；`SMA` 是通达信指数加权均值，
  权重 `m/n`。
- `RANK(x)` vs `TSRANK(x, n)`：截面 vs 时序，前者比的是同日不同股票，后者比的是
  同股票不同日。
- `TSRANK(x, n)` vs `TS_ARGMAX(x, n)`：前者给出当前值在窗口内的相对排名，后者给出
  窗口最大值出现的位置（1-based）。
- `LOG/SQRT` 对非正数会产生 `-inf/nan`，进入回归会让该截面被剔除——必要时先
  `x.where(x > 0)`。

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

### 3.1.6 _prepare_factor_values(self, winsorize=None, winsorize_param=3, standardize=False, industry_neutral=False, market_cap_neutral=False, industry_col="l1_code", market_cap_col="total_mv", market_cap_log=True, industry_data=None, adjust=True) -> pd.Series

功能：
- 对子类因子值执行预处理（复权输入、去极值、中性化、标准化），并按参数缓存结果。

流程：
1. 构造缓存键（含 `adjust`、中性化参数、`industry_data` 标识等）。
2. `adjust=True` 时先复权价格列后再调用 `calculate()`。
3. 按 date 截面去极值（可选）。
4. 行业/市值/行业+市值中性化（可选）。
5. 按 date 截面标准化（可选）。

日志口径：
- 参数缺失这类预期校验异常（如缺 `adj_factor`）走 `info`/`debug`，避免单测噪音。
- 非预期异常仍走 `logger.exception`。

返回：
- 预处理后的因子 Series。

### 3.1.7 get_clean_factor_and_forward_returns(self, quantiles=5, period=1, winsorize=None, winsorize_param=3, standardize=False, industry_neutral=False, market_cap_neutral=False, industry_col="l1_code", market_cap_col="total_mv", market_cap_log=True, industry_data=None, adjust=True) -> pd.DataFrame

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

`adjust` 语义：
- `True`：因子计算前先复权价格列，且 `next_ret` 使用 `close*adj_factor`。
- `False`：因子计算与 `next_ret` 都使用原始价格。

返回：
- DataFrame，核心列：factor、close、next_ret、factor_quantile。

### 3.1.8 calculate_daily_ic(self, result: pd.DataFrame) -> pd.Series

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
- 对于相邻两日持仓集合 `prev_set` 与 `cur_set`：
- 换手率 = `1 - |prev_set ∩ cur_set| / max(len(prev_set), len(cur_set))`
- 首个交易日为 `NaN`（无前一日可比）。
- 若中间某日该组为空集合，再次出现时会按与空集合比较计算（通常为 `1.0`）。

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
- 计算因子整体换手率（排序稳定性代理指标）。

定义：
- 因子换手率 = 1 - 截面秩自相关。

实现细节：
- 先转宽表并做截面百分位秩。
- 与前一日秩做 Spearman 相关。
- 常量截面自动记为 NaN。
- 该指标并非实际交易换手率：当秩相关为负时结果可能大于 1（理论上可到 2）。

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

### 3.1.19 calculate_net_returns_with_cost(self, group_ret: pd.DataFrame, group_turnover: pd.DataFrame, cost_bps: float = 10, buy_cost_bps: float | None = None, sell_cost_bps: float | None = None, round_trip_multiplier: float = 1.0) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]

功能：
- 按换手率扣交易成本，输出净收益。

参数：
- cost_bps：基础单边成本，单位 bps（兼容旧参数）。
- buy_cost_bps：买入单边成本，单位 bps；不传时回退到 `cost_bps`。
- sell_cost_bps：卖出单边成本，单位 bps；不传时回退到 `cost_bps`。
- round_trip_multiplier：双边成本倍率，默认 `1.0`。

返回：
1. 分组净收益 DataFrame
2. 多空毛收益 Series
3. 多空净收益 Series
4. 多空净绩效 Series

公式：
- 默认双边成本率 = `2 * cost_bps / 10000`
- 细粒度双边成本率 = `((buy_cost_bps + sell_cost_bps) / 10000) * round_trip_multiplier`
- 分组净收益 = 分组毛收益 - 分组换手率 × 双边成本率
- 多空净收益 = 多空毛收益 - (高组换手率 + 低组换手率) × 双边成本率

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
- `cost_bps` 用于图例展示，建议传双边等效成本（bps）。

### 3.1.31 create_factor_analysis_report(self, quantiles=5, period=1, winsorize=None, winsorize_param=3, standardize=False, transaction_cost_bps=10, transaction_buy_cost_bps=None, transaction_sell_cost_bps=None, transaction_round_trip_multiplier=1.0, industry_neutral=False, market_cap_neutral=False, industry_col="l1_code", market_cap_col="total_mv", market_cap_log=True, industry_data=None, adjust=True, output_dir="./output") -> Path

功能：
- 一键生成完整分析流程和 HTML 报告。

核心步骤：
1. 构建因子数据集。
2. 计算分组收益、多空收益、IC、换手率、交易成本扣减后的净收益。
3. 汇总统计指标并构建图表页面。
4. 生成可视化页面并写入 `output_dir/{factor.name}因子分析报告.html`。
5. 在控制台输出关键统计表。

关键参数：
- transaction_cost_bps：基础单边交易成本（bps）。
- transaction_buy_cost_bps / transaction_sell_cost_bps：可选买卖单边成本（bps）。
- transaction_round_trip_multiplier：双边成本倍率（默认 1.0）。
- adjust：因子计算前与远期收益计算是否统一使用复权价。

返回：
- 报告文件路径 `Path`。

## 4. 主程序示例（factor_engine.py 中）

示例子类 `my_factor` 的因子定义为：
- `MA(CLOSE, 20) / CLOSE - 1`。

示例流程：
1. 读取 stocks_daily.parquet。
2. 筛选时间区间并设置 MultiIndex。
3. 读取 industry.parquet（用于行业中性化）。
4. 调用 create_factor_analysis_report（`adjust=True` 时因子侧和收益侧统一复权）。

## 5. 术语与口径说明

1. 单边成本
- 指一次买入或卖出单独收取的成本。
- 报告净收益计算默认按双边扣减（买卖各收一次）。
- 若传 `transaction_buy_cost_bps/transaction_sell_cost_bps`，则优先使用细粒度成本模型。

2. bps
- 1 bps = 0.01%。
- 10 bps = 0.10%。

3. 因子换手率
- 用截面秩的日度自相关衡量因子稳定性，定义为 `1 - Spearman(rank_t, rank_{t-1})`。
- 该指标通常在 `[0,1]`，但排序显著反转时可大于 `1`，不应直接当作真实交易换手率。

4. 分组换手率
- 基于分组成分股集合变化计算，反映组合调仓强度。

## 6. 调用示例片段

以下示例均为最小可运行片段，默认在项目根目录执行。

### 6.1 config 模块

```python
from qlfactor import load_config, setup_logging

setup_logging()
cfg = load_config()
print(cfg.db_path)
print(type(cfg.pro))
```

### 6.2 Download（推荐 from_config）

```python
from qlfactor import Download, load_config, setup_logging

setup_logging()
cfg = load_config()
dl = Download.from_config(cfg)
print(dl.db_path)
```

### 6.3 Download.calendar / stocks_info / industry / stocks_daily / merge

```python
from qlfactor import Download, load_config, setup_logging

setup_logging()
cfg = load_config()
dl = Download.from_config(cfg)
dl.calendar()
dl.stocks_info()
dl.industry()
dl.stocks_daily()
dl.merge()
```

### 6.4 Factor 子类最小实现（对应 calculate）

```python
import pandas as pd
from qlfactor import Factor, load_config, setup_logging

setup_logging()
cfg = load_config()

class DemoFactor(Factor):
  def calculate(self):
    # 公式写法：无需在表达式里写 self.
    return self.FORMULA("MA(CLOSE, 20) / CLOSE - 1")

data = pd.read_parquet(cfg.db_path / "stocks_daily.parquet")
data = data.set_index(["date", "symbol"]).sort_index()
f = DemoFactor("demo", data)
print(f.calculate().dropna().head())
```

### 6.5 FORMULA / BIND 示例

```python
class DemoFactor2(Factor):
  def calculate(self):
    # 方式1：字符串公式
    f1 = self.FORMULA("EMA(CLOSE, 12) - EMA(CLOSE, 26)")

    # 方式2：先绑定再调用
    MA, CLOSE, RANK = self.BIND("MA", "CLOSE", "RANK")
    f2 = RANK(CLOSE / MA(CLOSE, 20))

    return f1 + f2
```

### 6.6 通达信风格函数示例

```python
# 近20日最高/最低区间位置
pos = (f.HHV(f.CLOSE, 20) - f.CLOSE) / (f.HHV(f.CLOSE, 20) - f.LLV(f.CLOSE, 20))

# 上穿信号（布尔序列）
signal = f.CROSS(f.MA(f.CLOSE, 5), f.MA(f.CLOSE, 20))

# 最近5天至少3天上涨
up_days = f.COUNT(f.CLOSE > f.REF(f.CLOSE, 1), 5)
cond = up_days >= 3
```

### 6.7 winsorize_3sigma / winsorize_mad / standardize_zscore

```python
factor_raw = f.calculate().dropna()

sample = factor_raw.groupby(level="date").head(200)
w1 = f.winsorize_3sigma(sample, n=3)
w2 = f.winsorize_mad(sample, k=3)
z = f.standardize_zscore(sample)

print(w1.head())
print(w2.head())
print(z.head())
```

### 6.8 _prepare_factor_values

```python
prepared = f._prepare_factor_values(
  winsorize="3sigma",
  winsorize_param=3,
  standardize=True,
  adjust=True,
)
print(prepared.dropna().head())
```

### 6.9 get_clean_factor_and_forward_returns

```python
result = f.get_clean_factor_and_forward_returns(
  quantiles=5,
  period=1,
  winsorize="3sigma",
  winsorize_param=3,
  standardize=True,
  adjust=True,
)
print(result.head())
```

### 6.10 calculate_daily_ic / ic_statistics_analysis / ic_mean_t_test

```python
daily_ic = f.calculate_daily_ic(result)
ic_stats = f.ic_statistics_analysis(daily_ic)
ic_t = f.ic_mean_t_test(daily_ic)

print(daily_ic.dropna().head())
print(ic_stats)
print(ic_t)
```

### 6.11 calculate_daily_group_ret / calculate_long_short / performance_analysis / calculate_all_group_performance

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

### 6.12 calculate_group_turnover / calculate_group_turnover_stats

```python
group_turnover = f.calculate_group_turnover(result)
group_turnover_stats = f.calculate_group_turnover_stats(group_turnover)

print(group_turnover.head())
print(group_turnover_stats)
```

### 6.13 calculate_factor_turnover_rate / factor_turnover_stats

```python
factor_values = f._prepare_factor_values(
  winsorize="3sigma",
  standardize=True,
  adjust=True,
).dropna()
factor_turnover = f.calculate_factor_turnover_rate(factor_values)
factor_turnover_stats = f.factor_turnover_stats(factor_turnover)

print(factor_turnover.dropna().head())
print(factor_turnover_stats)
```

### 6.14 calculate_turnover_return_correlation

```python
corr_series = f.calculate_turnover_return_correlation(group_turnover, group_ret)
print(corr_series)
```

### 6.15 calculate_net_returns_with_cost

```python
group_net_ret, ls_gross_ret, ls_net_ret, ls_net_perf = f.calculate_net_returns_with_cost(
  group_ret,
  group_turnover,
  cost_bps=10,
  buy_cost_bps=8,
  sell_cost_bps=12,
  round_trip_multiplier=1.5,
)

print(group_net_ret.head())
print(ls_gross_ret.head())
print(ls_net_ret.head())
print(ls_net_perf)
```

### 6.16 图表函数最小调用

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
chart10 = f.plot_long_short_net_vs_gross(ls_gross_ret, ls_net_ret, cost_bps=30)  # 双边等效 bps

for c in [chart1, chart2, chart3, chart4, chart5, chart6, chart7, chart8, chart9, chart10]:
  print(type(c))
```

### 6.17 create_factor_analysis_report

```python
report_path = f.create_factor_analysis_report(
  quantiles=5,
  period=1,
  winsorize="3sigma",
  winsorize_param=3,
  standardize=True,
  transaction_cost_bps=10,
  transaction_buy_cost_bps=8,
  transaction_sell_cost_bps=12,
  transaction_round_trip_multiplier=1.5,
  adjust=True,
)

print(f"report done: {report_path}")
```
