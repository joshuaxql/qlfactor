"""qlfactor —— A 股因子研究工具包。"""

from importlib import metadata

from qlfactor.config import Config, load_config, setup_logging
from qlfactor.download import Download
from qlfactor.factor_engine import Factor

try:
    __version__ = metadata.version("qlfactor")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = [
    "Config",
    "Download",
    "Factor",
    "__version__",
    "load_config",
    "setup_logging",
]
