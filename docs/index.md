# qlfactor

`qlfactor` 是一个面向 A 股日线数据的因子研究工具包，覆盖两条主线：

- 数据下载与整理：通过 Tushare 拉取交易日历、股票基础信息、行业成分、个股日线，并落地为 Parquet。
- 因子分析引擎：完成分组收益、IC、换手率、交易成本扣减与多空表现分析，输出 HTML 报告。

## 核心特性

- `src` 布局，支持作为库导入，也支持 CLI 命令。
- 环境配置强校验：`TUSHARE_TOKEN`、`DB_NAME` 缺一即报错。
- 因子公式 DSL：支持 `FORMULA("MA(CLOSE, 20) / CLOSE - 1")` 写法。
- 远期收益支持 `adjust=True/False`，`True`（默认）使用 `close * adj_factor`，`False` 使用原始 `close`。
- 报告自动输出到 `output_dir / "{factor.name}因子分析报告.html"`。

## 适用场景

- 快速搭建 A 股因子研究原型。
- 统一下载与存储日线数据（Parquet）。
- 对单因子做全链路评估并产出可视化报告。

## 项目结构

```text
qlfactor/
├─ src/qlfactor/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ download.py
│  ├─ factor_engine.py
│  └─ cli.py
├─ tests/
├─ docs/
└─ mkdocs.yml
```

## 文档导航

- [快速开始](getting-started.md)
- [API 参考](API.md)

## 下一步

先阅读 [快速开始](getting-started.md)，按“安装 → 配置 `.env` → 下载数据 → 运行最小因子示例”的顺序跑通一次完整流程。
