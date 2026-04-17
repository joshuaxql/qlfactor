from config import pro, db_path
import duckdb
import pandas as pd
from datetime import datetime
from tqdm import tqdm


class Download:
    def __init__(self, pro, db_path: str) -> None:
        self.pro = pro
        self.db_path = db_path
        self.con = duckdb.connect(db_path + "stocks_data.duckdb")

    def ts_change(self, ts_code: str) -> str:
        if ts_code.endswith(".SZ"):
            return "sz" + ts_code[:6]
        elif ts_code.endswith(".SH"):
            return "sh" + ts_code[:6]
        elif ts_code.endswith(".ICS"):
            return "ics" + ts_code[:6]
        elif ts_code.endswith(".BJ"):
            return "bj" + ts_code[:6]
        else:
            raise (f"{ts_code}格式错误")

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
                        df,  # 左表：行情数据
                        name_df,  # 右表：名称数据
                        left_on="trade_date",  # 左表匹配键：交易日期
                        right_on="start_date",  # 右表匹配键：名称生效日期
                        by="ts_code",  # 按股票代码分组匹配（多股票必备）
                        direction="backward",  # 向前匹配：找最近的生效日期
                    )
                    industry_df = pro.index_member_all(ts_code=stock)
                    df["industry"] = industry_df["l1_name"].iloc[0]
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
                            "industry",
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
                    symbol = self.ts_change(stock)
                    df["symbol"] = symbol
                    self.con.execute(
                        f"CREATE TABLE IF NOT EXISTS {symbol} AS SELECT * FROM df"
                    )
                    break
                except Exception as e:
                    print(e)
                    continue


if __name__ == "__main__":
    download = Download(pro, db_path)
    download.stocks_daily()
