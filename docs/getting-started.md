# 快速开始

本指南按“安装 -> 配置 -> 下载 -> 因子分析”顺序，带你跑通 qlfactor 的最小闭环。

## 1. 环境要求

- Python `>=3.12`
- 可用的 Tushare Pro Token

## 2. 安装
从PyPi（推荐）:
```bash
pip install qlfactor
```

开发模式：

```bash
uv pip install -e ".[dev]"
# 或
pip install -e ".[dev]"
```

只安装运行依赖：

```bash
pip install -e .
```

安装后可直接使用 CLI 命令 `qlfactor ...`。

## 3. 配置 `.env`

在项目根目录创建 `.env`（可复制 `.env.example`）：

```env
TUSHARE_TOKEN=YOUR_TUSHARE_TOKEN
DB_NAME=./data
```

注意：

- `TUSHARE_TOKEN` 与 `DB_NAME` 均为必填，缺失时 `load_config()` 立即抛 `ValueError`。
- `DB_NAME` 支持相对路径、绝对路径和 `~`，会被规范化为绝对路径。
- `DB_NAME` 目录不存在时会自动创建。

## 4. 下载数据（CLI）

推荐顺序：

```bash
qlfactor download calendar
qlfactor download stocks_info
qlfactor download industry
qlfactor download stocks_daily
qlfactor download merge
```

或一步执行：

```bash
qlfactor download all
```

也可以不安装脚本入口，直接模块运行（要求当前 Python 能导入 `qlfactor`，例如已 `pip install -e .`）：

```bash
python -m qlfactor.cli download all
```

`stocks_daily()` 特性：

- 单只股票最多重试 3 次。
- 某只股票失败只会记录 warning 并继续，最终可能得到“部分成功”的数据集。

## 5. 数据输出位置

下载后默认会在 `DB_NAME` 对应目录下生成：

- `calendar.parquet`
- `stocks_info.parquet`
- `industry.parquet`
- `stocks_daily/{symbol}.parquet`
- `stocks_daily.parquet`

## 6. 最小因子示例（库方式）

```python
from datetime import datetime
import pandas as pd

from qlfactor import Factor, load_config, setup_logging


class MA20Factor(Factor):
    def calculate(self):
        return self.FORMULA("MA(CLOSE, 20) / CLOSE - 1")


setup_logging()
cfg = load_config()

data = pd.read_parquet(cfg.db_path / "stocks_daily.parquet")
data = data[data["date"].between(datetime(2020, 1, 1), datetime(2020, 12, 31))]
data = data.set_index(["date", "symbol"]).sort_index()

industry_data = pd.read_parquet(cfg.db_path / "industry.parquet")

factor = MA20Factor("ma20", data)
report_path = factor.create_factor_analysis_report(
    quantiles=5,
    period=1,
    winsorize="3sigma",
    standardize=True,
    industry_neutral=True,
    market_cap_neutral=True,
    industry_data=industry_data,
    adjust=True,
    output_dir="./output",
)

print(report_path)
```

## 7. 输入数据约束

- `Factor` 输入必须是 `MultiIndex(date, symbol)`。
- 在进入引擎前建议执行 `.sort_index()`。
- 当 `adjust=True`（默认）时，数据必须包含 `adj_factor` 列，否则会抛 `KeyError`。

## 8. 日志与报告

- `setup_logging()` 默认同时输出到终端和 `debug.log`。
- 报告默认输出到 `./output/{factor.name}因子分析报告.html`。

## 9. 验证与构建

运行单元测试：

```bash
./.venv/Scripts/python -m unittest discover -s tests
```

构建发行包：

```bash
uv build
```

产物位于 `dist/`（`.whl` 与 `.tar.gz`）。
