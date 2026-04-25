"""配置加载与日志初始化测试。"""

import logging
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import qlfactor.config as cfg_mod
from qlfactor.config import Config, load_config, setup_logging


class TestConfigDataclass(unittest.TestCase):
    def test_str_db_path_normalized_to_absolute_path(self):
        c = Config(token="t", db_path="./relative/dir")
        self.assertIsInstance(c.db_path, Path)
        self.assertTrue(c.db_path.is_absolute())

    def test_path_db_path_passes_through(self):
        p = Path("./xx").resolve()
        c = Config(token="t", db_path=p)
        self.assertEqual(c.db_path, p)


class TestLoadConfig(unittest.TestCase):
    def setUp(self) -> None:
        # 隔离环境，避免外部 .env 干扰
        self._saved = {k: os.environ.get(k) for k in ("TUSHARE_TOKEN", "DB_NAME")}
        os.environ.pop("TUSHARE_TOKEN", None)
        os.environ.pop("DB_NAME", None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_missing_token_raises(self):
        os.environ["DB_NAME"] = "./tmp_qlfactor_db"
        with self.assertRaises(ValueError) as ctx:
            load_config(env_file="/nonexistent/.env", create_db_dir=False)
        self.assertIn("TUSHARE_TOKEN", str(ctx.exception))

    def test_missing_db_name_raises(self):
        os.environ["TUSHARE_TOKEN"] = "fake"
        with self.assertRaises(ValueError) as ctx:
            load_config(env_file="/nonexistent/.env", create_db_dir=False)
        self.assertIn("DB_NAME", str(ctx.exception))

    def test_load_with_required_envs(self):
        os.environ["TUSHARE_TOKEN"] = "fake"
        os.environ["DB_NAME"] = "./tmp_qlfactor_db"
        try:
            c = load_config(env_file="/nonexistent/.env", create_db_dir=True)
            self.assertEqual(c.token, "fake")
            self.assertTrue(c.db_path.is_absolute())
            self.assertTrue(c.db_path.exists())
        finally:
            try:
                Path("./tmp_qlfactor_db").resolve().rmdir()
            except OSError:
                pass


class TestSetupLogging(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_state = cfg_mod._LOGGING_CONFIGURED
        self._prev_handlers = list(logging.getLogger().handlers)

    def tearDown(self) -> None:
        cfg_mod._LOGGING_CONFIGURED = self._prev_state
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in self._prev_handlers:
            root.addHandler(h)

    def test_force_reinstalls_handlers(self):
        cfg_mod._LOGGING_CONFIGURED = False
        setup_logging(level=logging.WARNING, log_file=None, force=True)
        root = logging.getLogger()
        # 至少有 StreamHandler
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in root.handlers))
        before = len(root.handlers)

        # 第二次默认调用不应叠加
        setup_logging(level=logging.WARNING, log_file=None)
        self.assertEqual(len(root.handlers), before)


if __name__ == "__main__":
    unittest.main()
