import requests
from bs4 import BeautifulSoup
import csv
import matplotlib.pyplot as plt
import numpy as np

def fetch_html():
    url = "https://datachart.500.com/ssq/history/newinc/outball.php?start=00000&end=26041"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://datachart.500.com/"
    }

    res = requests.get(url, headers=headers)
    res.encoding = "gbk"

    return res.text


def parse_html(html):
    soup = BeautifulSoup(html, "html.parser")

    rows = soup.select("tr")

    data = []

    for row in rows:
        tds = row.find_all("td")
        print(f" tds : {[td.text.strip() for td in tds]}")

        # 必须是完整数据行
        if len(tds) != 15:
            continue

        issue = tds[0].text.strip()

        # ✅ 关键：直接按位置取
        reds_order = [tds[i].text.strip() for i in range(2, 8)]

        blue = tds[14].text.strip()

        data.append([issue] + reds_order + [blue])

    return data


def save_csv(data, csv_file="ssq_raw.csv"):
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # writer.writerow([
        #     "期号",
        #     "红1(出球)", "红2", "红3",
        #     "红4", "红5", "红6",
        #     "蓝球"
        # ])

        writer.writerows(data)

def parse_data(csv_file):
    data = []

    raw_red_balls = []
    raw_blue_balls = []
    normal_red_balls = []
    normal_blue_balls = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            # row = ['26041','08','10',...]
            date = row[0]
            reds = list(map(int, row[1:7]))
            blue = int(row[7])

            data.append({
                "date": date,
                "reds": reds,
                "blue": blue
            })

            raw_red_balls.append(reds)
            raw_blue_balls.append( blue )
            normal_red_balls.append([ 2 * (red - 1)/ 32 - 1 for red in reds])
            normal_blue_balls.append( 2 * (blue - 1)/ 15 - 1 )
            print(f"date: {date}, reds: {reds}, blue: {blue}")

    # check data
    assert len(raw_red_balls) == len(raw_blue_balls), "red and blue balls count mismatch"
    assert len(raw_red_balls) == len(normal_blue_balls), "red and blue balls count mismatch"

    for i in range(len(normal_blue_balls)):
        normal_blue_ball = normal_blue_balls[i]
        raw_blue_ball = raw_blue_balls[i]
        recover_blue_ball = int(round((normal_blue_ball + 1) * (16 - 1)/2 + 1))
        if recover_blue_ball != raw_blue_ball:
            print(f"Blue ball not compared: normal {recover_blue_ball}, raw {raw_blue_ball}")
            break

    for i in range(len(normal_red_balls)):
        for j in range(len(normal_red_balls[i])):
            normal_red_ball = normal_red_balls[i][j]
            raw_red_ball = raw_red_balls[i][j]
            recover_red_ball = (normal_red_ball + 1) * (33 - 1)/2 + 1
            if recover_red_ball != raw_red_ball:
                print(f"Red ball not compared: normal {recover_red_ball}, raw {raw_red_ball}")
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
    # plt.savefig('data/blue_balls_distribution.png', dpi=300) 

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
    # plt.savefig('data/red_balls_distribution.png', dpi=300)

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
        cmap='hsv',   
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


def main():
    print("> [INFO]: Acquire outball data ...")
    html = fetch_html()
    data = parse_html(html)
    print(f"> [INFO]: Total get {len(data)} items.")

    csv_file = "data/ball_raw.csv"
    save_csv(data, csv_file)
    parse_data(csv_file)

    print("> [INFO]: All done.")


if __name__ == "__main__":
    main()