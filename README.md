# qlfactor

一个基于 A 股日线数据的因子研究项目，包含两条主线：

- 数据下载与整理：通过 Tushare 拉取交易日历、股票基础信息、日线行情并落地为 Parquet。
- 因子分析引擎：完成因子分组收益、IC 分析、换手率分析、交易成本扣减与多空净值分析。

## 1. 当前能力

### 1.1 数据侧

- 下载交易日历（SSE 开市日）
- 下载两市股票基础信息（上市/退市/暂停）
- 下载单只股票日线、复权因子、日线估值与换手指标
- 合并历史证券简称（namechange）
- 合并所有个股文件为总表 stocks_daily.parquet

### 1.2 因子分析侧

- 去极值：3sigma、MAD
- 标准化：截面 Z-Score
- 远期收益计算与分位分组
- IC、IC 累加、IC 分布、T 检验
- 分组收益、多空组合收益
- 分组换手率、因子整体换手率
- 换手率分位统计（P25/P50/P75）
- 换手率与收益相关性
- 按换手率扣交易成本后的净收益分析
- 输出 HTML 可视化报告

## 2. 项目结构

```text
qlfactor/
├─ data/
│  ├─ stocks_daily/
│  └─ stocks_data.duckdb
├─ output/
│  └─ 因子分析报告.html
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ download.py
│  └─ factor_engine.py
├─ pyproject.toml
├─ requirement.md
├─ README.md
└─ API.md
```

## 3. 环境准备

### 3.1 Python 版本

- 建议 Python 3.12+

### 3.2 安装依赖

项目使用 pyproject 管理依赖，建议使用 uv 或 pip。

示例（pip）：

```bash
pip install -U pyarrow pyecharts python-dotenv scipy tushare tqdm
```

说明：当前代码实际使用了 tqdm，请确保环境中已安装。

### 3.3 配置环境变量

在项目根目录创建 .env：

```env
TUSHARE_TOKEN=你的tushare_token
DB_NAME=./data/
```

含义：

- TUSHARE_TOKEN：Tushare Pro Token
- DB_NAME：数据目录（默认 ./data/）

## 4. 数据下载流程

入口文件：src/download.py

Download 类提供 4 个步骤：

1. calendar：下载交易日历
2. stocks_info：下载股票基础信息
3. stocks_daily：逐股票下载日线与扩展指标
4. merge：合并为总表

推荐顺序：

```python
from config import pro, db_path
from download import Download

d = Download(pro, db_path)
d.calendar()
d.stocks_info()
d.stocks_daily()
d.merge()
```

## 5. 因子分析流程

入口文件：src/factor_engine.py

你需要定义一个继承 Factor 的子类，并实现 caculate 方法返回 MultiIndex(date, symbol) 的因子值 Series。

示例（项目内置）：

```python
class ma(Factor):
	def caculate(self):
		return (
			self.data["turnover_rate"]
			.groupby(level="symbol")
			.transform(lambda x: x.rolling(20).mean() / x)
		)
```

然后调用：

```python
ma_factor.create_factor_analysis_report(
	winsorize="3sigma",
	standardize=True,
	transaction_cost_bps=10,
)
```

## 6. 关键参数说明

- quantiles：分组数，默认 5
- period：远期收益周期（天），默认 1
- winsorize：去极值方式，可选 3sigma 或 mad
- winsorize_param：去极值参数
- standardize：是否做截面 Z-Score
- transaction_cost_bps：单边交易成本（bps）

### 6.1 单边 10bps 是什么

- 1 bps = 0.01% = 0.0001
- 10 bps = 0.10% = 0.001
- 单边表示一次交易动作（买入或卖出）收一次成本

在本项目里，transaction_cost_bps=10 表示：

- 分组净收益 = 分组毛收益 - 分组换手率 × 0.001
- 多空净收益 = 多空毛收益 - (多头换手率 + 空头换手率) × 0.001

## 7. 输出结果

### 7.1 控制台输出

- 分组绩效指标
- 多空组合指标
- IC 统计与 T 检验
- 分组换手率统计（含分位数）
- 因子换手率统计（含分位数）
- 换手率与收益相关系数
- 扣成本后的分组净绩效、多空净绩效

### 7.2 报告文件

- output/因子分析报告.html

报告包含累计收益、IC 图、换手率图、相关性图、毛净收益对比图等。

## 8. API 文档

完整函数文档见：API.md

## 9. 注意事项

- Tushare 接口有频率和积分限制，批量下载时建议分批执行。
- 首次全量下载耗时较长，建议先缩小时间区间调试。
- stocks_daily 下载过程可能因网络抖动失败，代码内已做重试。
- 因子计算请保证索引与价格数据严格对齐（date, symbol）。
