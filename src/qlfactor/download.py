"""数据下载：基于 Tushare 拉取 A 股交易日历、基础信息、日线、行业成分，落地为 Parquet。"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from qlfactor.config import Config

logger = logging.getLogger(__name__)


class Download:
    """Tushare 数据下载器。

    Examples:
        >>> from qlfactor import Download, load_config, setup_logging
        >>> setup_logging()
        >>> cfg = load_config()
        >>> dl = Download.from_config(cfg)
        >>> dl.calendar()
        >>> dl.stocks_info()
        >>> dl.stocks_daily()
        >>> dl.merge()
    """

    DAILY_BASE_COLUMNS = [
        "symbol",
        "date",
        "name",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "volume",
        "amount",
        "adj_factor",
    ]

    DAILY_BASIC_COLUMNS = [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ]

    DAILY_OUTPUT_COLUMNS = DAILY_BASE_COLUMNS + DAILY_BASIC_COLUMNS

    def __init__(self, pro: Any, db_path: Path | str) -> None:
        """构造下载器。

        Args:
            pro: Tushare Pro 客户端（``ts.pro_api(token)`` 的返回值）。
            db_path: 数据目录，自动转 :class:`Path` 并按需创建。
        """
        self.pro = pro
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Config) -> "Download":
        """便捷工厂：从 :class:`Config` 构造。"""
        return cls(config.pro, config.db_path)

    def _safe_concat(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """拼接前清理空表/全 NA 列，规避 pandas FutureWarning。"""
        valid_frames = []
        for frame in frames:
            if frame is None or frame.empty:
                continue
            cleaned = frame.dropna(axis=1, how="all")
            if cleaned.empty:
                continue
            valid_frames.append(cleaned)

        if len(valid_frames) == 0:
            return pd.DataFrame()
        return pd.concat(valid_frames, ignore_index=True, sort=False)

    def calendar(self) -> None:
        """下载 SSE 交易日历（开市日）→ ``calendar.parquet``。"""
        logger.info("下载交易日历")
        calendar = self.pro.trade_cal(
            exchange="SSE",
            is_open="1",
            start_date="19000101",
            end_date="20500101",
            fields="cal_date",
        )
        calendar["cal_date"] = pd.to_datetime(calendar["cal_date"], format="%Y%m%d")
        calendar = calendar.iloc[::-1]
        calendar.to_parquet(self.db_path / "calendar.parquet", index=False)
        logger.info("交易日历已写入：%s", self.db_path / "calendar.parquet")

    def stocks_info(self) -> None:
        """下载 SSE/SZSE 全部股票（L/D/P 三种状态）→ ``stocks_info.parquet``。"""
        logger.info("下载股票基础信息")
        frames = [
            self.pro.stock_basic(exchange=exchange, list_status=status)
            for exchange in ("SSE", "SZSE")
            for status in ("L", "D", "P")
        ]
        stocks_info = self._safe_concat(frames)
        stocks_info.drop(columns=["symbol"], inplace=True)
        stocks_info.rename(columns={"ts_code": "symbol"}, inplace=True)
        stocks_info["list_date"] = pd.to_datetime(
            stocks_info["list_date"], format="%Y%m%d"
        )
        stocks_info.to_parquet(self.db_path / "stocks_info.parquet", index=False)
        logger.info("股票基础信息条数：%d", len(stocks_info))

    def industry(self, level: str = "L1", src: str = "SW2021") -> None:
        """下载申万行业成分变更表 → ``industry.parquet``。"""
        logger.info("下载行业分类：level=%s, src=%s", level, src)
        classify = self.pro.index_classify(level=level, src=src)
        if classify is None or classify.empty:
            logger.warning("index_classify 返回空")
            pd.DataFrame().to_parquet(self.db_path / "industry.parquet", index=False)
            return

        codes = classify["index_code"].dropna().unique().tolist()
        members: list[pd.DataFrame] = []

        pbar = tqdm(codes, desc="industry")
        for code in pbar:
            pbar.set_description(f"industry {code}")
            err: Exception | None = None
            ok = False
            for _ in range(3):
                try:
                    part_y = self.pro.index_member_all(l1_code=code, is_new="Y")
                    part_n = self.pro.index_member_all(l1_code=code, is_new="N")
                    part = pd.concat([part_y, part_n])
                    if part is None or part.empty:
                        ok = True
                        break
                    part = part.copy()
                    part["query_l1_code"] = code
                    members.append(part)
                    ok = True
                    break
                except Exception as e:
                    err = e
                    continue

            if not ok:
                logger.error("行业成分下载失败 %s：%s", code, err)

        data = self._safe_concat(members)
        if data.empty:
            logger.warning("无可用行业成分数据")
            data.to_parquet(self.db_path / "industry.parquet", index=False)
            return

        for col in ("in_date", "out_date"):
            if col in data.columns:
                data[col] = pd.to_datetime(
                    data[col], format="%Y%m%d", errors="coerce"
                )

        sort_cols = [
            col
            for col in ("l1_code", "l2_code", "l3_code", "ts_code", "in_date")
            if col in data.columns
        ]
        if sort_cols:
            data = data.sort_values(sort_cols).reset_index(drop=True)

        data.to_parquet(self.db_path / "industry.parquet", index=False)
        logger.info("行业成分数据条数：%d", len(data))

    def _fetch_stock_daily_chunks(
        self, stock: str, calendar_list: list[datetime], basic: bool = True
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        for j in range(0, len(calendar_list), 5000):
            chunk = calendar_list[j : j + 5000]
            start_date = chunk[0].strftime("%Y%m%d")
            end_date = chunk[-1].strftime("%Y%m%d")

            daily_df = self.pro.daily(
                ts_code=stock, start_date=start_date, end_date=end_date
            )
            adj_df = self.pro.adj_factor(
                ts_code=stock, start_date=start_date, end_date=end_date
            )
            chunk_df = pd.merge(daily_df, adj_df, "left", ["ts_code", "trade_date"])

            if basic:
                basic_df = self.pro.daily_basic(
                    ts_code=stock, start_date=start_date, end_date=end_date
                )
                chunk_df = pd.merge(
                    chunk_df,
                    basic_df,
                    "left",
                    ["ts_code", "trade_date", "close"],
                )

            chunk_df = chunk_df.iloc[::-1]
            if chunk_df.empty or chunk_df.dropna(how="all").empty:
                continue
            chunks.append(chunk_df)

        df = self._safe_concat(chunks)

        if not basic:
            for col in self.DAILY_BASIC_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA

        return df

    def _merge_stock_name(self, df: pd.DataFrame, stock: str) -> pd.DataFrame:
        name_df = (
            self.pro.namechange(ts_code=stock)
            .drop_duplicates(subset=["ts_code", "name", "start_date", "end_date"])
            .reset_index(drop=True)
        )
        name_df["start_date"] = pd.to_datetime(name_df["start_date"], format="%Y%m%d")
        name_df["end_date"] = pd.to_datetime(name_df["end_date"], format="%Y%m%d")
        name_df = name_df.sort_values("start_date")

        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("trade_date")
        df = pd.merge_asof(
            df,
            name_df,
            left_on="trade_date",
            right_on="start_date",
            by="ts_code",
            direction="backward",
        )
        return df

    def _format_stock_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                "ts_code": "symbol",
                "trade_date": "date",
                "vol": "volume",
            }
        )

        for col in self.DAILY_OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        return df[self.DAILY_OUTPUT_COLUMNS]

    def stocks_daily(self, basic: bool = True) -> None:
        """逐股下载日线 + 复权因子 + （可选）日级估值/换手指标。

        每只股票最多重试 3 次，失败仅记日志后继续，因此可能产出部分成功的数据集。
        """
        end = datetime.now()
        stocks_info = pd.read_parquet(self.db_path / "stocks_info.parquet")
        calendar = pd.read_parquet(self.db_path / "calendar.parquet")
        stocks_list = stocks_info["symbol"].to_list()
        list_dates = stocks_info["list_date"].to_list()

        out_dir = self.db_path / "stocks_daily"
        out_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(range(len(stocks_info)))
        for i in pbar:
            stock = stocks_list[i]
            for _ in range(3):
                try:
                    pbar.set_description(f"{stock}")
                    start = list_dates[i]
                    calendar_i = calendar[calendar["cal_date"].between(start, end)]
                    calendar_list = calendar_i["cal_date"].to_list()
                    if len(calendar_list) == 0:
                        break
                    df = self._fetch_stock_daily_chunks(
                        stock, calendar_list, basic=basic
                    )
                    df = self._merge_stock_name(df, stock)
                    df = self._format_stock_daily(df)
                    df.to_parquet(out_dir / f"{stock}.parquet", index=False)
                    break
                except Exception as e:
                    logger.warning("下载失败 %s：%s", stock, e)
                    continue

    def merge(self) -> None:
        """合并 ``stocks_daily/*.parquet`` 为单文件 ``stocks_daily.parquet``。"""
        files_dir = self.db_path / "stocks_daily"
        if not files_dir.exists():
            logger.error("目录不存在：%s（先跑 stocks_daily()）", files_dir)
            return

        files = [str(p) for p in files_dir.glob("*.parquet")]
        if not files:
            logger.warning("无 parquet 文件可合并：%s", files_dir)
            return

        with ProcessPoolExecutor() as executor:
            dfs = list(
                tqdm(
                    executor.map(pd.read_parquet, files),
                    total=len(files),
                    desc="Reading",
                )
            )

        dfs = [df for df in dfs if not df.empty and not df.dropna(how="all").empty]
        if len(dfs) == 0:
            logger.warning("无有效 parquet 数据")
            return

        data = self._safe_concat(dfs)
        data = data.sort_values(["date", "symbol"])
        out_path = self.db_path / "stocks_daily.parquet"
        data.to_parquet(out_path, index=False)
        logger.info("合并完成：%s（%d 行）", out_path, len(data))


if __name__ == "__main__":
    from qlfactor.config import load_config, setup_logging

    setup_logging()
    cfg = load_config()
    dl = Download.from_config(cfg)
    # dl.calendar()
    # dl.stocks_info()
    # dl.industry()
    # dl.stocks_daily()
    # dl.merge()
