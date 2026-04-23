from config import pro, db_path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor


class Download:
    def __init__(self, pro, db_path: str) -> None:
        self.pro = pro
        self.db_path = db_path

    def calendar(self) -> None:
        calendar = pro.trade_cal(
            exchange="SSE",
            is_open="1",
            start_date="19000101",
            end_date="20500101",
            fields="cal_date",
        )
        calendar["cal_date"] = pd.to_datetime(calendar["cal_date"], format="%Y%m%d")
        calendar = calendar.iloc[::-1]
        calendar.to_parquet(self.db_path + "calendar.parquet", index=False)

    def stocks_info(self) -> None:
        SSE_stocks_info_L = pro.stock_basic(exchange="SSE", list_status="L")
        SSE_stocks_info_D = pro.stock_basic(exchange="SSE", list_status="D")
        SSE_stocks_info_P = pro.stock_basic(exchange="SSE", list_status="P")
        SZSE_stocks_info_L = pro.stock_basic(exchange="SZSE", list_status="L")
        SZSE_stocks_info_D = pro.stock_basic(exchange="SZSE", list_status="D")
        SZSE_stocks_info_P = pro.stock_basic(exchange="SZSE", list_status="P")
        stocks_info = pd.concat(
            [
                SSE_stocks_info_L,
                SSE_stocks_info_D,
                SSE_stocks_info_P,
                SZSE_stocks_info_L,
                SZSE_stocks_info_D,
                SZSE_stocks_info_P,
            ],
            ignore_index=True,
        )
        stocks_info.drop(columns=["symbol"], inplace=True)
        stocks_info.rename(columns={"ts_code": "symbol"}, inplace=True)
        stocks_info["list_date"] = pd.to_datetime(
            stocks_info["list_date"], format="%Y%m%d"
        )
        stocks_info.to_parquet(self.db_path + "stocks_info.parquet", index=False)

    def stocks_daily(self) -> None:
        end = datetime.now()
        stocks_info = pd.read_parquet(self.db_path + "stocks_info.parquet")
        calendar = pd.read_parquet(self.db_path + "calendar.parquet")
        stocks_list = stocks_info["symbol"].to_list()
        list_dates = stocks_info["list_date"].to_list()
        pbar = tqdm(range(len(stocks_info)))
        for i in pbar:
            for _ in range(3):
                try:
                    stock = stocks_list[i]
                    pbar.set_description(f"{stock}")
                    start = list_dates[i]
                    calendar_i = calendar[calendar["cal_date"].between(start, end)]
                    calendar_list = calendar_i["cal_date"].to_list()
                    df = pd.DataFrame()
                    for j in range(0, len(calendar_list), 5000):
                        chunk = calendar_list[j : j + 5000]
                        start_date = chunk[0].strftime("%Y%m%d")
                        end_date = chunk[-1].strftime("%Y%m%d")
                        temp = self.pro.daily(
                            ts_code=stock,
                            start_date=start_date,
                            end_date=end_date,
                        )
                        adj_temp = self.pro.adj_factor(
                            ts_code=stock,
                            start_date=start_date,
                            end_date=end_date,
                        )
                        basic_temp = self.pro.daily_basic(
                            ts_code=stock,
                            start_date=start_date,
                            end_date=end_date,
                        )
                        temp = pd.merge(
                            temp, adj_temp, "left", ["ts_code", "trade_date"]
                        )
                        temp = pd.merge(
                            temp, basic_temp, "left", ["ts_code", "trade_date", "close"]
                        )
                        temp = temp.iloc[::-1]
                        df = pd.concat([df, temp], ignore_index=True)
                    name_df = (
                        self.pro.namechange(ts_code=stock)
                        .drop_duplicates(
                            subset=["ts_code", "name", "start_date", "end_date"]
                        )
                        .reset_index(drop=True)
                    )
                    name_df["start_date"] = pd.to_datetime(
                        name_df["start_date"], format="%Y%m%d"
                    )
                    name_df["end_date"] = pd.to_datetime(
                        name_df["end_date"], format="%Y%m%d"
                    )
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
                    df = df.rename(
                        columns={
                            "ts_code": "symbol",
                            "trade_date": "date",
                            "vol": "volume",
                        }
                    )
                    df = df[
                        [
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
                    ]
                    df.to_parquet(
                        self.db_path + f"stocks_daily/{stock}.parquet", index=False
                    )
                    break
                except Exception as e:
                    print(e)
                    continue

    def merge(self) -> None:
        files = [
            os.path.join(db_path + "stocks_daily", f)
            for f in os.listdir(db_path + "stocks_daily")
            if f.endswith(".parquet")
        ]

        with ProcessPoolExecutor() as executor:
            dfs = list(
                tqdm(
                    executor.map(pd.read_parquet, files),
                    total=len(files),
                    desc="Reading",
                )
            )

        data = pd.concat(dfs, ignore_index=True)
        data = data.sort_values(["date", "symbol"])
        data.to_parquet(db_path + "stocks_daily.parquet", index=False)


if __name__ == "__main__":
    download = Download(pro, db_path)
    # download.calendar()
    # download.stocks_info()
    # download.stocks_daily()
    download.merge()
