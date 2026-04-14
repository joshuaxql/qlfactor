from dotenv import load_dotenv
import tushare as ts
import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="debug.log",
)
# 自动加载 .env 文件
load_dotenv()

# 获取环境变量
token = os.getenv("TUSHARE_TOKEN")
db_path = os.getenv("DB_NAME", "./data/")

if token and db_path:
    pro = ts.pro_api(token)
    logging.info(f"已从.env读取到token:{token}, db_path:{db_path}")
else:
    raise ValueError("请先在.env设置环境变量")
