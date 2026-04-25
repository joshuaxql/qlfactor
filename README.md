# qlfactor

一个基于 A 股日线数据的因子研究工具包，包含两条主线：

- **数据下载与整理**：通过 Tushare 拉取交易日历、股票基础信息、日线行情、行业成分，并落地为 Parquet。
- **因子分析引擎**：完成因子分组收益、IC 分析、换手率分析、交易成本扣减、多空净值分析，输出 HTML 报告。

## 1. 当前能力

### 1.1 数据侧

- 下载交易日历（SSE 开市日）
- 下载两市股票基础信息（上市/退市/暂停）
- 下载单只股票日线、复权因子、日级估值与换手指标
- 合并历史证券简称（namechange）
- 合并所有个股文件为总表 `stocks_daily.parquet`
- 下载申万行业成分变更表（SW2021）

### 1.2 因子分析侧

- 去极值：3sigma、MAD
- 标准化：截面 Z-Score
- 行业 / 市值 / 行业+市值中性化
- **远期收益可选是否复权**（`adjust=True/False`）
- 分位分组、IC、IC 累加、IC 分布、T 检验
- 分组收益、多空组合收益、换手率、相关性
- 按换手率扣交易成本后的净收益分析
- 输出 HTML 可视化报告

## 2. 项目结构

```text
qlfactor/
├─ pyproject.toml
├─ README.md
├─ API.md
├─ src/
│  └─ qlfactor/            # PyPI 包目录（src layout）
│     ├─ __init__.py
│     ├─ config.py
│     ├─ download.py
│     ├─ factor_engine.py
│     └─ cli.py
├─ tests/                  # unittest 套件
└─ data/                   # .gitignore，由 DB_NAME 指定
```

## 3. 安装

### 3.1 从源码（开发）

```bash
uv pip install -e ".[dev]"
# 或
pip install -e ".[dev]"
```

### 3.2 从 PyPI

```bash
pip install qlfactor
```

### 3.3 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```env
TUSHARE_TOKEN=你的tushare_token
DB_NAME=./data
```

两者均为**必填**：缺失时 `load_config()` 直接抛 `ValueError`。`DB_NAME` 不存在的目录会自动创建，结尾是否带斜杠都可以——内部统一用 `pathlib.Path`。

## 4. 数据下载

### 4.1 命令行

安装后可用 `qlfactor` 命令：

```bash
qlfactor download calendar          # 下载交易日历
qlfactor download stocks_info       # 下载股票基础信息
qlfactor download industry          # 下载申万行业成分
qlfactor download stocks_daily      # 下载逐股日线
qlfactor download merge             # 合并为总表 stocks_daily.parquet
qlfactor download all               # 顺序执行以上 5 步
qlfactor --log-level DEBUG download stocks_daily   # 调日志级别
```

### 4.2 库形式

```python
from qlfactor import Download, load_config, setup_logging

setup_logging()              # 终端 + debug.log 同时输出
cfg = load_config()
dl = Download.from_config(cfg)

dl.calendar()
dl.stocks_info()
dl.industry()                # 行业中性化需要
dl.stocks_daily()
dl.merge()
```

下载顺序不能打乱（`stocks_daily` 与 `merge` 依赖前两步生成的 parquet）。`stocks_daily()` 对每只股票最多重试 3 次，失败仅记日志后继续，因此可能产出部分成功的数据集。

## 5. 因子分析

### 5.1 公式写法（推荐）

`Factor` 已内置常用字段别名与公式函数，可直接用 `FORMULA("MA(CLOSE, 20)")` 风格，不必在表达式里写 `self.`。

```python
from datetime import datetime
import pandas as pd
from qlfactor import Factor, load_config, setup_logging

setup_logging()
cfg = load_config()

class my_factor(Factor):
    def calculate(self):
        return self.FORMULA("MA(CLOSE, 20) / CLOSE - 1")

data = pd.read_parquet(cfg.db_path / "stocks_daily.parquet")
data = data[data["date"].between(datetime(2020, 1, 1), datetime(2020, 12, 31))]
data = data.set_index(["date", "symbol"]).sort_index()

industry_data = pd.read_parquet(cfg.db_path / "industry.parquet")

my_factor("ma20", data).create_factor_analysis_report(
    winsorize="3sigma",
    standardize=True,
    transaction_cost_bps=10,
    industry_neutral=True,
    market_cap_neutral=True,
    industry_data=industry_data,
    adjust=True,           # 远期收益使用 close*adj_factor 复权
)
```

如果不想写字符串，也可先绑定后调用：

```python
class my_factor(Factor):
    def calculate(self):
        MA, CLOSE = self.BIND("MA", "CLOSE")
        return MA(CLOSE, 20) / CLOSE - 1
```

### 5.2 已内置字段（大写）

`OPEN, HIGH, LOW, CLOSE, PRE_CLOSE, CHANGE, PCT_CHG, VOLUME, AMOUNT, ADJ_FACTOR,
TURNOVER_RATE, TURNOVER_RATE_F, VOLUME_RATIO, PE, PE_TTM, PB, PS, PS_TTM,
DV_RATIO, DV_TTM, TOTAL_SHARE, FLOAT_SHARE, FREE_SHARE, TOTAL_MV, CIRC_MV`

### 5.3 已内置公式函数

- 基础：`MA, EMA, RANK, REF, DELTA, STD, SUM`
- 通达信风格：`ABS, MAX, MIN, IF, COUNT, EVERY, EXIST, HHV, LLV, TSRANK, SMA, CROSS`
- 数学：`LOG, EXP, SQRT`

## 6. 关键参数说明

- `quantiles`：分组数，默认 5
- `period`：远期收益周期（天），默认 1
- `winsorize`：去极值方式，可选 `3sigma` 或 `mad`
- `winsorize_param`：去极值参数（默认 3）
- `standardize`：是否做截面 Z-Score
- `industry_neutral` / `market_cap_neutral`：是否中性化
- `transaction_cost_bps`：单边交易成本（bps）
- `adjust`：远期收益是否使用复权价（默认 `True`，需要 `adj_factor` 列；若数据已在外部预先复权请传 `False`）
- `output_dir`：报告输出目录，默认 `./output`

### 6.1 单边 10bps 是什么

- 1 bps = 0.01% = 0.0001
- 10 bps = 0.10% = 0.001
- 单边表示一次交易动作（买入或卖出）收一次成本

具体扣减：

- 分组净收益 = 分组毛收益 − 分组换手率 × 0.001
- 多空净收益 = 多空毛收益 − (多头换手率 + 空头换手率) × 0.001

## 7. 输出结果

### 7.1 日志输出

`setup_logging()` 默认同时写到：

- 终端（StreamHandler）
- 仓库根目录的 `debug.log`（FileHandler）

分组绩效、IC、换手率等汇总都会以 `logging.info` 形式打印，所以同时进终端与文件。可通过 `setup_logging(level=...)` 或 `qlfactor --log-level DEBUG ...` 调级别。

### 7.2 报告文件

`create_factor_analysis_report()` 输出到 `output_dir / "{factor.name}因子分析报告.html"`。报告包含累计收益、IC 图、换手率图、相关性图、毛净收益对比图等。

## 8. API 文档

完整函数文档见 [API.md](API.md)。

## 9. 注意事项

- Tushare 接口有频率和积分限制，批量下载时建议分批执行。
- 首次全量下载耗时较长，建议先缩小时间区间调试。
- 因子计算请保证索引与价格数据严格对齐（`date`, `symbol`）。

## 10. 测试

```bash
./.venv/Scripts/python -m unittest discover -s tests
# 或
python -m unittest discover -s tests
```

测试覆盖公式函数、远期收益（含/不含复权）、IC、分组收益、绩效、去极值/标准化、配置加载与日志初始化。
