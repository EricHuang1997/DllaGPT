import json
import random
# 读取JSON文件
with open('C:/Users/Eric/PycharmProjects/ELLAM/Data_Directory/restaurant-multi-mix.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 获取"data"键的数据
items = data["data"]
random.shuffle(items)
# 计算70%的分割点
split_index = int(len(items) * 0.67)


# 分割数据
data_70_percent = items[:split_index]
data_30_percent = items[split_index:]

if __name__ == '__main__':
    # 将70%的数据写入新的JSON文件
    with open('C:/Users/Eric/PycharmProjects/ELLAM/Data_Directory/restaurant-multi-test.json', 'w', encoding='utf-8') as file:
        json.dump({"data": data_70_percent}, file, ensure_ascii=False, indent=4)

    # 将剩下的30%的数据写入另一个新的JSON文件
    with open('C:/Users/Eric/PycharmProjects/ELLAM/Data_Directory/restaurant-multi-val.json', 'w', encoding='utf-8') as file:
        json.dump({"data": data_30_percent}, file, ensure_ascii=False, indent=4)
