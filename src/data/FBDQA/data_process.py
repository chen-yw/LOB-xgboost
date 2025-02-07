# 生成5分类标签和回归标签
import pandas as pd
import os
def process_regression_label(data):
    '''生成回归标签标签'''
    mid_price = (data['n_ask1'] + data['n_bid1']) / 2 + 1
    data['r5'] = mid_price.shift(-5) / mid_price - 1
    data['r5'] = data['r5'].fillna(0)
    data['r10'] = mid_price.shift(-10) / mid_price - 1
    data['r10'] = data['r10'].fillna(0)
    data['r20'] = mid_price.shift(-20) / mid_price - 1
    data['r20'] = data['r20'].fillna(0)
    data['r40'] = mid_price.shift(-40) / mid_price - 1
    data['r40'] = data['r40'].fillna(0)
    data['r60'] = mid_price.shift(-60) / mid_price - 1
    data['r60'] = data['r60'].fillna(0)
    return data

def process_classification_label(data):
    '''生成分类标签'''
    data['l5'] = pd.cut(data['r5'], bins=[-0.01, -0.001, -0.0005, 0.0005, 0.001,0.01], labels=[0, 1, 2, 3, 4])
    data['l10'] = pd.cut(data['r10'], bins=[-0.01, -0.001, -0.0005, 0.0005, 0.001,0.01], labels=[0, 1, 2, 3, 4])
    data['l20'] = pd.cut(data['r20'], bins=[-0.01, -0.0015, -0.001, 0.0001, 0.0015,0.01], labels=[0, 1, 2, 3, 4])
    data['l40'] = pd.cut(data['r40'], bins=[-0.01, -0.0015, -0.001, 0.0001, 0.0015,0.01], labels=[0, 1, 2, 3, 4])
    data['l60'] = pd.cut(data['r60'], bins=[-0.01, -0.0015, -0.001, 0.0001, 0.0015,0.01], labels=[0, 1, 2, 3, 4])
    return data

def process_regression_data():
    '''将每只标的上午下午的数据分别处理，然后合并'''
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_path,'data')
    for i in range(10):
        print(f"Processing sym{i}")
        data = pd.DataFrame()
        for j in range(79):
            try:
                am = pd.read_csv(f"{file_path}/snapshot_sym{i}_date{j}_am.csv")
                pm = pd.read_csv(f"{file_path}/snapshot_sym{i}_date{j}_pm.csv")
                am = process_regression_label(am)
                pm = process_regression_label(pm)
                data_ = pd.concat([am, pm])
                data = pd.concat([data, data_])
            except:
                continue
        os.makedirs(f"{current_path}/regression", exist_ok=True)
        data.to_csv(f"{current_path}/regression/sym{i}.csv", index=False)

def process_classification_data():
    """将每只标的上午下午的数据分别处理，然后合并"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_path, 'data')
    for i in range(10):
        print(f"Processing sym{i}")
        data = pd.read_csv(f"{current_path}/regression/sym{i}.csv")
        data = process_classification_label(data)
        os.makedirs(f"{current_path}/classification", exist_ok=True)
        data.to_csv(f"{current_path}/classification/sym{i}.csv", index=False)

if __name__ == '__main__':
    process_regression_data()
    process_classification_data()