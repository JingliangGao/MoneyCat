import requests
import re
import csv
import time
import matplotlib.pyplot as plt
import numpy as np

# ====== 配置 ======
url = 'http://kaijiang.500.com/static/info/kaijiang/xml/ssq/list.xml?_A=BLWXUIYA1546584359929'
csv_file = 'data/ssq_data.csv'

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

# read data from csv
raw_red_balls = []
raw_blue_balls = []
normal_red_balls = []
normal_blue_balls = []
with open(csv_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        reds = row['red'].split(',')
        raw_reds_list = [int(red.strip()) for red in reds] 
        raw_red_balls.append(raw_reds_list)
        raw_blue_balls.append( int(row['blue']) )
        normal_red_balls.append([ (int(red.strip()) - 33)/33 + 0.5 for red in reds])
        normal_blue_balls.append( (int(row['blue']) - 16)/16 + 0.5 )

# check data
assert len(raw_red_balls) == len(raw_blue_balls), "red and blue balls count mismatch"
assert len(raw_red_balls) == len(normal_blue_balls), "red and blue balls count mismatch"

for i in range(len(normal_blue_balls)):
    normal_blue_ball = normal_blue_balls[i]
    raw_blue_ball = raw_blue_balls[i]
    if (normal_blue_ball - 0.5) * 16 + 16 != raw_blue_ball:
        print(f"Blue ball not compared: normal {normal_blue_ball}, raw {raw_blue_ball}")
        break

for i in range(len(normal_red_balls)):
    for j in range(len(normal_red_balls[i])):
        normal_red_ball = normal_red_balls[i][j]
        raw_red_ball = raw_red_balls[i][j]
        if (normal_red_ball - 0.5) * 33 + 33 != raw_red_ball:
            print(f"Red ball not compared: normal {normal_red_ball}, raw {raw_red_ball}")
            break   

# plot data distribution
plt.figure(figsize=(20, 4))  
plt.scatter(
    range(len(raw_blue_balls)),
    raw_blue_balls,
    color='blue',
    alpha=0.1,
    s=5,                     # point size
    label='Blue Balls'
)
plt.savefig('data/blue_balls_distribution.png', dpi=300) 

plt.figure(figsize=(20, 4))
for i in range(6):
    red_balls_i = [raw_red_balls[j][i] for j in range(len(raw_red_balls))]
    plt.scatter(
        range(len(red_balls_i)),
        red_balls_i,
        alpha=0.1,
        s=5,                 # point size
        label=f'Red Ball {i+1}'
    )
plt.savefig('data/red_balls_distribution.png', dpi=300)

# convert to numpy array
red_array = np.array(raw_red_balls)  # [3435, 6]
blue_array = np.array(raw_blue_balls).reshape(-1, 1)  # [3435, 1]
red_normalized_array = np.array(normal_red_balls)  # [3435, 6]
blue_normalized_array = np.array(normal_blue_balls).reshape(-1, 1)  # [3435, 1]
time_series = np.hstack((red_array, blue_array)).T  # [7, 3435]
normal_time_series = np.hstack((red_normalized_array, blue_normalized_array)).T  # [7, 3435]

# plot heatmap
plt.figure(figsize=(18, 4))
plt.imshow(
    time_series,
    aspect='auto',
    cmap='magma',   
    interpolation='bilinear'  # 平滑
)

plt.colorbar(label='Value')

plt.xlabel("Time")
plt.ylabel("number")
plt.xticks([])
plt.title("Red-Blue Ball Heatmap")
plt.savefig("data/ball_heatmap.png", dpi=300)

# save time_series to npy file
data_array_path = 'data/time_series.npy'
np.save(data_array_path, time_series)
print(f"save time_series in {data_array_path} with shape: {time_series.shape}")
np.save('data/normal_time_series.npy', normal_time_series)
print(f"save normal_time_series in data/normal_time_series.npy with shape: {normal_time_series.shape}")