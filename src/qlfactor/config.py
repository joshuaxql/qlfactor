"""配置加载与日志初始化。

设计原则
========

- **无 import 时副作用**：仅 ``load_config()`` / ``setup_logging()`` 才会读取
  ``.env``、构建 Tushare 客户端、安装 logging handler。
- ``TUSHARE_TOKEN`` 与 ``DB_NAME`` 都是 **必填**，缺一即抛 ``ValueError``。
- ``DB_NAME`` 用 :class:`pathlib.Path` 规范化，自动处理结尾 ``/`` 缺失、相对路径、
  ``~`` 展开等问题，并按需创建目录。
- 日志默认同时输出到 ``debug.log`` 文件与终端（stderr）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_LOGGING_CONFIGURED = False


@dataclass
class Config:
    """运行 qlfactor 所需的配置对象。

    Attributes:
        token: Tushare Pro token。
        db_path: 数据目录（已规范化为绝对 :class:`Path`）。
    """

    token: str
    db_path: Path
    _pro: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.db_path, Path):
            self.db_path = Path(self.db_path)
        self.db_path = self.db_path.expanduser().resolve()

    @property
    def pro(self) -> Any:
        """惰性创建 Tushare Pro 客户端，避免无网络环境下的 import 副作用。"""
        if self._pro is None:
            import tushare as ts

            self._pro = ts.pro_api(self.token)
        return self._pro


def load_config(
    env_file: str | os.PathLike[str] | None = None,
    *,
    create_db_dir: bool = True,
) -> Config:
    """从 ``.env`` 加载并校验配置。

    Args:
        env_file: 可选的 ``.env`` 路径；不传则按 :func:`dotenv.load_dotenv`
            的默认搜索路径查找。
        create_db_dir: 是否在缺失时创建 ``DB_NAME`` 指向的目录。

    Raises:
        ValueError: ``TUSHARE_TOKEN`` 或 ``DB_NAME`` 缺失。
    """
    if env_file is not None:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv()

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError(
            "TUSHARE_TOKEN 必须在 .env 中设置（或作为环境变量传入）"
        )

    raw_db_path = os.getenv("DB_NAME")
    if not raw_db_path:
        raise ValueError(
            "DB_NAME 必须在 .env 中设置——指向数据目录（如 ./data）"
        )

    config = Config(token=token, db_path=Path(raw_db_path))

    if create_db_dir:
        config.db_path.mkdir(parents=True, exist_ok=True)

    logger.info("已加载配置：db_path=%s", config.db_path)
    return config


def setup_logging(
    level: int = logging.INFO,
    log_file: str | os.PathLike[str] | None = "debug.log",
    *,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force: bool = False,
) -> None:
    """初始化日志：默认同时写入文件与终端。

    幂等——重复调用不会叠加 handler，除非传 ``force=True``。

    Args:
        level: 根 logger 级别。
        log_file: 文件日志路径；传 ``None`` 关闭文件输出。
        fmt: 日志格式串。
        force: 是否清空已有 handler 后重建。
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return

    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    formatter = logging.Formatter(fmt)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    for h in handlers:
        h.setFormatter(formatter)
        root.addHandler(h)

    root.setLevel(level)
    _LOGGING_CONFIGURED = True
