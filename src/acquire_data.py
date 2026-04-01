import requests
import re
import csv
import time

# ====== 配置 ======
url = 'http://kaijiang.500.com/static/info/kaijiang/xml/ssq/list.xml?_A=BLWXUIYA1546584359929'
csv_file = 'ssq_data.csv'

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
}

# ====== 请求数据 ======
try:
    response = requests.get(url, headers=headers, timeout=10)
    response.encoding = 'utf-8'
    text = response.text
except Exception as e:
    print("请求失败:", e)
    exit()

# ====== 正则解析 ======
# <row opencode="10,11,12,13,26,28|11" opentime="2003-02-23 00:00:00" ... />
pattern = re.compile(r'<row.*?opencode="(.*?)".*?opentime="(.*?)"', re.S)
ssq_data = pattern.findall(text)

if not ssq_data:
    print("没有抓取到数据")
    exit()

print(f"共抓取 {len(ssq_data)} 条双色球数据")

# ====== 保存 CSV ======
# CSV 列名：期号, 红球, 蓝球, 开奖时间
with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['red', 'blue', 'date'])  # 写入表头
    
    for data in ssq_data:
        info, date = data
        if '|' in info:
            red, blue = info.split('|')
        else:
            red, blue = info, ''
        
        writer.writerow([red, blue, date])

print(f"数据已保存到 {csv_file}")