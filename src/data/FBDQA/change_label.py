# 此文件用于修改标签
# 原本的标签是5,10 tick采用的是万五为分界 20,40,60采用的是千一为分界。
# 考虑到我们的目标是做到千三的盈利，所以一律改成千三为分界

import os
import pandas as pd

directory = "ML_data_300"
new_directory = "ML_data_300_with_new_label"

def process_csv_files(directory,new_directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否是 .csv 文件
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(new_directory, filename)
            print(file_path)
            # 读取 .csv 文件
            df = pd.read_csv(file_path)
            # 在这里处理读取到的数据
            ask_1 = df.iloc[:,0]
            bid_1 = df.iloc[:,5]
            price = (ask_1 + bid_1) / 2
            df.iloc[:,-5] = label_data(price,5,0.003)
            df.iloc[:,-4] = label_data(price,10,0.003)
            df.iloc[:,-3] = label_data(price,20,0.003)
            df.iloc[:,-2] = label_data(price,40,0.003)
            df.iloc[:,-1] = label_data(price,60,0.003)
            df.to_csv(new_file_path, index=False)

def label_data(price_seires,period,threshold):
    label = []
    for i in range(len(price_seires)):
        if i < len(price_seires) - period:
            if price_seires[i + period] - price_seires[i] > threshold:
                label.append(2)
            elif price_seires[i + period] - price_seires[i] < -threshold:
                label.append(0)
            else:
                label.append(1)
        else:
            label.append(1)
    return label
            
process_csv_files(directory,new_directory)
