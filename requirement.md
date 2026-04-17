# 股票因子分析

目前的进度：
1.完成了基础的数据下载
2.完成了基础的因子分析引擎

待完成：
1.在数据下载代码里增加股票历史名称和行业数据
2.补全因子分析引擎的换手率分析

# 一些接下来会用到的tushare接口

申万行业成分构成(分级)
接口：index_member_all
描述：按三级分类提取申万行业成分，可提供某个分类的所有成分，也可按股票代码提取所属分类，参数灵活
限量：单次最大2000行，总量不限制
权限：用户需2000积分可调取，积分获取方法请参阅积分获取办法



输入参数

名称	类型	必选	描述
l1_code	str	N	一级行业代码
l2_code	str	N	二级行业代码
l3_code	str	N	三级行业代码
ts_code	str	N	股票代码
is_new	str	N	是否最新（默认为“Y是”）


输出参数

名称	类型	默认显示	描述
l1_code	str	Y	一级行业代码
l1_name	str	Y	一级行业名称
l2_code	str	Y	二级行业代码
l2_name	str	Y	二级行业名称
l3_code	str	Y	三级行业代码
l3_name	str	Y	三级行业名称
ts_code	str	Y	成分股票代码
name	str	Y	成分股票名称
in_date	str	Y	纳入日期
out_date	str	Y	剔除日期
is_new	str	Y	是否最新Y是N否


接口示例


#获取黄金分类的成份股
df = pro.index_member_all(l3_code='850531.SI')

#获取000001.SZ所属行业
df = pro.index_member_all(ts_code='000001.SZ')


数据示例

      l1_code l1_name     l2_code       l2_name  l3_code     l3_name    ts_code       name   in_date
0   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  000506.SZ      *ST中润  20220729
1   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  001337.SZ       四川黄金  20230224
2   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  600988.SH       赤峰黄金  20040414
3   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  600489.SH       中金黄金  20030812
4   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  600547.SH       山东黄金  20030826
5   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  002155.SZ       湖南黄金  20070815
6   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  002237.SZ       恒邦股份  20080428
7   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  601069.SH       西部黄金  20150115
8   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  000975.SZ       银泰黄金  20190724
9   801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  300139.SZ       晓程科技  20220729
10  801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  600687.SH   退市刚泰(退市)  20130701
11  801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  600807.SH       济南高新  20220729
12  801050.SI    有色金属  801053.SI     贵金属  850531.SI      黄金  600311.SH  *ST荣华(退市)  20140102

---

股票曾用名
接口：namechange
描述：历史名称变更记录

输入参数

名称	类型	必选	描述
ts_code	str	N	TS代码
start_date	str	N	公告开始日期
end_date	str	N	公告结束日期
输出参数

名称	类型	默认输出	描述
ts_code	str	Y	TS代码
name	str	Y	证券名称
start_date	str	Y	开始日期
end_date	str	Y	结束日期
ann_date	str	Y	公告日期
change_reason	str	Y	变更原因
接口示例


pro = ts.pro_api()

df = pro.namechange(ts_code='600848.SH', fields='ts_code,name,start_date,end_date,change_reason')
数据样例

    ts_code    name    start_date   end_date      change_reason
0  600848.SH   上海临港   20151118      None         改名
1  600848.SH   自仪股份   20070514  20151117         撤销ST
2  600848.SH   ST自仪     20061026  20070513         完成股改
3  600848.SH   SST自仪   20061009  20061025        未股改加S
4  600848.SH   ST自仪     20010508  20061008         ST
5  600848.SH   自仪股份  19940324  20010507         其他