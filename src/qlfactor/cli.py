"""``qlfactor`` 命令行入口（PyPI ``console_scripts``）。

仅暴露最常用的下载子命令——因子分析以库形式调用更合适。
"""

from __future__ import annotations

import argparse
import logging
import sys

from qlfactor.config import load_config, setup_logging
from qlfactor.download import Download

logger = logging.getLogger("qlfactor.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlfactor",
        description="A 股因子研究工具 —— 数据下载 / 因子分析",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help=".env 路径，默认按 dotenv 默认搜索",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="终端与文件的日志级别",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="数据下载")
    dl.add_argument(
        "step",
        choices=["calendar", "stocks_info", "industry", "stocks_daily", "merge", "all"],
        help="下载步骤；all 顺序执行 calendar→stocks_info→industry→stocks_daily→merge",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(level=getattr(logging, args.log_level))
    cfg = load_config(args.env_file)

    if args.command == "download":
        dl = Download.from_config(cfg)
        steps_all = ["calendar", "stocks_info", "industry", "stocks_daily", "merge"]
        steps = steps_all if args.step == "all" else [args.step]
        for step in steps:
            logger.info("执行下载步骤：%s", step)
            getattr(dl, step)()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
