from config import pro, db_path
import pandas as pd
from datetime import datetime
from tqdm import tqdm


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
        calendar.to_parquet(self.db_path + "calendar.parquet")

    def stocks_info(self) -> None:
        stocks_info_L = pro.stock_basic(list_status="L")
        stocks_info_D = pro.stock_basic(list_status="D")
        stocks_info_P = pro.stock_basic(list_status="P")
        stocks_info = pd.concat(
            [stocks_info_L, stocks_info_D, stocks_info_P], ignore_index=True
        )
        stocks_info.drop(columns=["symbol"], inplace=True)
        stocks_info.rename(columns={"ts_code": "symbol"}, inplace=True)
        stocks_info["list_date"] = pd.to_datetime(
            stocks_info["list_date"], format="%Y%m%d"
        )
        stocks_info.to_parquet(self.db_path + "stocks_info.parquet")

    def stocks_daily(self) -> None:
        end = datetime.now()
        stocks_info = pd.read_parquet(self.db_path + "stocks_info.parquet")
        calendar = pd.read_parquet(self.db_path + "calendar.parquet")
        stocks_list = stocks_info["symbol"].to_list()
        list_dates = stocks_info["list_date"].to_list()
        for i in tqdm(range(4500, len(stocks_info))):
            start = list_dates[i]
            calendar_i = calendar[calendar["cal_date"].between(start, end)]
            calendar_list = calendar_i["cal_date"].to_list()
            df = pd.DataFrame()
            for j in range(0, len(calendar_list), 5000):
                chunk = calendar_list[j : j + 5000]
                temp = self.pro.daily(
                    ts_code=stocks_list[i],
                    start_date=chunk[0].strftime("%Y%m%d"),
                    end_date=chunk[-1].strftime("%Y%m%d"),
                )
                adj_temp = self.pro.adj_factor(
                    ts_code=stocks_list[i],
                    start_date=chunk[0].strftime("%Y%m%d"),
                    end_date=chunk[-1].strftime("%Y%m%d"),
                )
                basic_temp = self.pro.daily_basic(
                    ts_code=stocks_list[i],
                    start_date=chunk[0].strftime("%Y%m%d"),
                    end_date=chunk[-1].strftime("%Y%m%d"),
                )
                temp = pd.merge(temp, adj_temp, "left", ["ts_code", "trade_date"])
                temp = pd.merge(
                    temp, basic_temp, "left", ["ts_code", "trade_date", "close"]
                )
                temp = temp.iloc[::-1]
                df = pd.concat([df, temp], ignore_index=True)
            df = df.rename(
                columns={"ts_code": "symbol", "trade_date": "date", "vol": "volume"}
            )
            df["date"] = pd.to_datetime(df["date"])
            df.to_parquet(self.db_path + f"daily/{stocks_list[i]}.parquent")


if __name__ == "__main__":
    download = Download(pro, db_path)
    download.stocks_daily()
