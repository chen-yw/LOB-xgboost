from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,fbeta_score
import numpy as np
from datetime import datetime
import os
import pandas as pd
import xgboost as xgb
from loguru import logger
import sys
import random
import string

current_dir = os.path.dirname(__file__) # 获取当前文件的目录
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) # 获取上一级目录的路径
log_file_path = os.path.join(current_dir, 'log', 'out.log') # 设置日志文件路径为上一级目录中的 log 文件夹下的 out.log


logger.add(log_file_path, level='INFO', rotation='10 MB', format='{time} | {level} | {message}')

def find_save_path(experiment_id):
    """
    Find the directory path for saving results associated with a given `experiment_id`. This function constructs a directory path within the
    'log/results' directory relative to the script's location.

    :param experiment_id: model identifier, (str).
    :return: directory path, (str).
    """
    root_path = parent_dir
    dir_path = os.path.join(root_path, "log", "results", experiment_id)
    return dir_path

def generate_id(name, target_stock):
    """
    Generate a unique experiment identifier based on the input `name` and the current timestamp in the format "YYYY-MM-DD_HH_MM_SS".
    Create a directory path using this identifier within the 'loggers/results'  directory relative to the script's location, and if
    it doesn't exist, create it.

    :param name: name of the DL model to be used in the experiment, (str).
    :return: experiment_id: unique experiment identifier, (str).
    """
    random_string_part = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(7))
    init_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_id = f"{target_stock}_{name}_{init_time}_{random_string_part}"

    root_path = sys.path[0]
    dir_path = f"{root_path}/log/results/{experiment_id}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return experiment_id

# 需要写一个计算平均pnl的东西出来
def cal_average_pnl(label,ask_1,bid_1,window = 20):
    # print("start to cal_average_pnl")
    price = ask_1 + bid_1
    price = price/2
    trade_count_0 = 0
    total_pnl_0 = 0 # 上面是预测标签为0的交易，对应下跌
    trade_count_2 = 0
    total_pnl_2 = 0 # 上面是预测标签为2的交易，对应上涨
    for i in range(len(label)-20):
        if int(label[i]) == 0:
            trade_count_0 += 1
            total_pnl_0 += (price.iloc[i] - price.iloc[i+window])/(1+price.iloc[i])
        if int(label[i]) == 2:
            trade_count_2 += 1
            total_pnl_2 += (price.iloc[i+window] - price.iloc[i])/(1+price.iloc[i])
    total_pnl_0 *= 10000
    total_pnl_2 *= 10000
    logger.info("Trade count for label 0: {}".format(trade_count_0))
    logger.info("Trade count for label 2: {}".format(trade_count_2))
    if trade_count_0 == 0 or trade_count_2 == 0:
        logger.info("No trade for label 0 or label 2")
        return
    logger.info("Average pnl for label 0: {}".format(total_pnl_0/trade_count_0))
    logger.info("Average pnl for label 2: {}".format(total_pnl_2/trade_count_2))
    logger.info("total average pnl: {}".format((total_pnl_0+total_pnl_2)/(trade_count_0+trade_count_2)))
    

def data_loader(stock_id = 0,train_ratio=0.8,label_number = 0):
    base_path = "data/FBDQA/ML_data_300/"
    file_name = f"stock_{stock_id}_features.csv"
    file_name = os.path.join(base_path, file_name)
    all_datasets = pd.read_csv(file_name)
    data = all_datasets.iloc[:,:-5]
    all_target = all_datasets.iloc[:,-5:]
    # label = all_target.iloc[:,label_number]
    label = all_target
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    train_size = int(len(data)*train_ratio)
    x_train = data.iloc[:train_size]
    y_train = label.iloc[:train_size]
    x_test = data.iloc[train_size:]
    y_test = label.iloc[train_size:]
    return x_train, y_train, x_test, y_test

def data_loader_for_many_stocks(stock_id_1 = 0,stock_id_2 = 1,train_ratio = 0.8,label_numer = 0):
    base_path = "data/FBDQA/ML_data_300/"
    file_name = f"stock_{stock_id_1}_features.csv"
    file_name = ""
    file_name = os.path.join(base_path,file_name)
    all_datasets = pd.read_csv(file_name)
    
    data = all_datasets.iloc[:,:-5]
    all_target = all_datasets.iloc[:,-5:]
    label = all_target.iloc[:,label_number]
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)
    train_size = int(len(data)*train_ratio)
    x_train = data.iloc[:train_size]
    y_train = label.iloc[:train_size]
    x_test = data.iloc[train_size:]
    y_test = label.iloc[train_size:]
    
    for stock_id in range(stock_id_1+1,stock_id_2):
        file_name = f"stock_{stock_id_1}_features.csv"
        file_name = os.path.join(base_path, file_name)
        all_datasets = pd.read_csv(file_name)
        tmp_all_target = all_datasets.iloc[:,:-5]
        tmp_label = tmp_all_target.iloc[:,label_numer]
        tmp_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        tmp_data.fillna(data.mean(), inplace=True)
        train_size = int(len(data)*train_ratio)
        tmp_x_train = data.iloc[:train_size]
        tmp_y_train = label.iloc[:train_size]
        tmp_x_test = data.iloc[train_size:]
        tmp_y_test = label.iloc[train_size:]
        
        x_train = pd.concat(x_train,tmp_x_train)
        y_train = pd.concat(y_train,tmp_y_train)
        x_test = pd.concat(x_test,tmp_x_test)
        y_test = pd.concat(y_test,tmp_y_test)
    return x_train,y_train,x_test,y_test    
        
params_default = {
    'booster': 'gbtree',
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softmax',
    'num_class': 3,  # 假设有 3 个类别
    'eval_metric': 'mlogloss'
}

def train(x_train, y_train, params=params_default,stock_id = 0,label_number = 0, num_round = 100,my_weights = [1,1,1]):
    weights = np.ones_like(y_train)
    
    # 类别不平衡处理，根据不同类别的样本数量进行不同权重
    weights[y_train == 0] = my_weights[0]
    weights[y_train == 1] = my_weights[1]
    weights[y_train == 2] = my_weights[2]
    
    dtrain = xgb.DMatrix(x_train, y_train,weight = weights)
    bst = xgb.train(params, dtrain, num_round)
    # base_path = "E:/vscode_file/python/FBDQA/1/LOB-Predict/train/XGBoost/models"
    # new_dir = os.path.join(base_path, f"stock_{stock_id}_label_{label_number}")
    # if not os.path.exists(new_dir):
    #     os.makedirs(new_dir)
    # bst.save_model(os.path.join(new_dir,'xgboost_model.json'))
    return bst
    
def predict(x_test, model, threshold=0.7):
    dtest = xgb.DMatrix(x_test)
    probabilities = model.predict(dtest, output_margin=False)
    
    # 根据阈值调整预测结果
    adjusted_predictions = []
    # print(probabilities)
    for i, probs in enumerate(probabilities):
        # print(i,probs)
        if probs[0] > threshold:
            adjusted_predictions.append(0)
        elif probs[2] > threshold:
            adjusted_predictions.append(2)
        else:
            adjusted_predictions.append(1)
    
    return np.array(adjusted_predictions)

def predict_with_possibility(x_test,model):
    dtest = xgb.DMatrix(x_test)
    return model.predict(dtest,output_margin=False)

def test(all_targets,all_predictions):
    my_accuracy_score = accuracy_score(all_targets, all_predictions)
    my_classification_report = classification_report(all_targets, all_predictions, digits=4)
    cm = confusion_matrix(all_targets, all_predictions)
    f_0_5_score = fbeta_score(all_targets, all_predictions, beta=0.5,average = None)
    f_1_score = fbeta_score(all_targets, all_predictions,beta=1,average = None)
    
    custon_weight = [0.5,0,0.5]
    
    weighted_f_0_5_score = sum(f_0_5_score * custon_weight)
    weighted_f_1_score = sum(f_1_score * custon_weight)
    
    logger.success('\naccuracy_score:{}'.format( my_accuracy_score))
    logger.success("\n{}".format(my_classification_report))
    logger.success('\nconfusion_matrix:\n{}'.format(cm))
    logger.success('\nf_0_5_score:{}'.format(f_0_5_score))
    logger.success('\nf_1_score:{}'.format(f_1_score))
    logger.success('\nweighted_f_0_5_score:{}'.format(weighted_f_0_5_score))
    logger.success('\nweighted_f_1_score:{}\n'.format(weighted_f_1_score))
    

