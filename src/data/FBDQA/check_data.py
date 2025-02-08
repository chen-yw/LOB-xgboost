# 此文件用来检查一下更改标签之后的数据
import os
import pandas as pd

directory = "ML_data_300"
new_directory = "ML_data_300_with_new_label"

def process_csv_files(directory,new_directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(new_directory):
        # 检查文件是否是 .csv 文件
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(new_directory, filename)
            df = pd.read_csv(file_path)
            new_df = pd.read_csv(new_file_path)
            array = df.iloc[:,-3] 
            new_array = new_df.iloc[:,-3]
            count_0 = (array == 0).sum()
            count_1 = (array == 1).sum()
            count_2 = (array == 2).sum()
            new_count_0 = (new_array == 0).sum()
            new_count_1 = (new_array == 1).sum()
            new_count_2 = (new_array == 2).sum()
            print(f"old label 0 count: {count_0}, new label 0 count: {new_count_0}")
            print(f"old label 1 count: {count_1}, new label 1 count: {new_count_1}")
            print(f"old label 2 count: {count_2}, new label 2 count: {new_count_2}")
            print()

process_csv_files(directory,new_directory)
