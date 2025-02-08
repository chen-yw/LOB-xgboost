import xgboost as xgb
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import logger
from utils import utils
from loguru import logger

params_default = {
    'booster': 'gbtree',
    'eta': 0.1,
    'max_depth':6, # 最大树高为2的时候效果最佳
    'min_child_weight': 1, # 最小子节点权重为1的时候效果最佳
    'gamma': 0,
    'subsample': 0.8, # 0.8时最佳
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': 3,  # 假设有 3 个类别
    'eval_metric': 'mlogloss'
}

num_round_default = 100 

def train_and_test(stock_id = 1,train_ratio=0.8,label_number = 0,params = params_default,num_round = 100,threshold = 0.7,weights = [1,1,1]):
    print("数据开始加载")
    logger.success(f"\nstock_id:{stock_id}, label_number:{label_number}, num_round:{num_round}, threshold={threshold}")
    x_train,y_train,x_test,y_test = utils.data_loader(stock_id = stock_id,train_ratio=train_ratio,label_number = label_number)
    print("数据加载完毕")
    y_train = y_train.iloc[:,label_number]
    y_test = y_test.iloc[:,label_number]
    weights = weights
    model = utils.train(x_train,y_train,params = params,stock_id = stock_id,label_number = label_number,num_round = num_round,my_weights = weights)
    print("模型训练完毕")
    y_predict = utils.predict(x_test,model,threshold=threshold)
    new_y_predict = utils.predict_with_possibility(x_test,model)
    utils.test(y_test,y_predict)
    ask_1 = x_test.iloc[:,0]
    bid_1 = x_test.iloc[:,5]
    utils.cal_average_pnl(y_predict,ask_1,bid_1)
    print("模型测试完毕\n------------------------------------\n")

train_and_test(stock_id=1,label_number=3,train_ratio=0.8,threshold=0.45,weights=[1,1,1]) 

# logger.success("Start to search for best hyperparameter!")
# for stock_id in range(0,10,1):
#     for label_number in range(3,4,1):
#         for max_depth in range(2,7,1):
#             params_default['max_depth'] = max_depth
#             for threshold in np.arange(0.3,0.55,0.05):
#                     logger.success(f"\nstock_id:{stock_id}, label_number:{label_number}, max_depth:{max_depth}, threshold:{threshold}")
#                     train_and_test(stock_id=stock_id,label_number=label_number,threshold=threshold,weights=[1.2,1,1])

def test_for_cal_avergae_pnl(stock_id = 1,train_raio=0.8,label_number = 0,params = params_default,num_round = 100,weights = [1,1,1]):
    print(f"stock_id:{stock_id}, label_number:{label_number},num_round={num_round}")
    print("数据开始加载")
    logger.success(f"\nstock_id:{stock_id}, label_number:{label_number}, num_round:{num_round}")
    x_train,y_train,x_test,y_test = utils.data_loader(stock_id = stock_id,train_ratio=train_ratio,label_number = label_number)
    y_train = y_train.iloc[:,label_number]
    y_test = y_test.iloc[:,label_number]
    print("数据加载完毕")
    model = xgb.Booster()
    model.load_model("model.json")
    y_predict = utils.predict(x_test,model)
    utils.test(y_test,y_predict)
    ask_1 = x_test.iloc[:,0]
    bid_1 = x_test.iloc[:,5]
    utils.cal_average_pnl(y_predict,ask_1,bid_1)
    print("模型测试完毕\n------------------------------------\n")
     
# test_for_cal_avergae_pnl(stock_id = 7,label_number=2)
    
def test_for_output_possibility(stock_id = 1,train_ratio=0.8,label_number = 0,params = params_default,num_round = 100,weights = [1,1,1]):
    print(f"stock_id:{stock_id}, label_number:{label_number},num_round={num_round}")
    print("数据开始加载")
    logger.success(f"\nstock_id:{stock_id}, label_number:{label_number}, num_round:{num_round}")
    x_train,y_train,x_test,y_test = utils.data_loader(stock_id = stock_id,train_ratio=train_ratio,label_number = label_number)
    y_train = y_train.iloc[:,label_number]
    y_test = y_test.iloc[:,label_number]
    print("数据加载完毕")
    model = xgb.Booster()
    model.load_model("model.json")
    
    y_predict_1 = utils.predict(x_test,model)
    print(y_predict_1)
    y_predict = utils.predict_with_possibility(x_test,model)
    print(y_predict)
    
    
    # y_predict = utils.predict(x_test,model)
    # utils.test(y_test,y_predict)
    # ask_1 = x_test.iloc[:,0]
    # bid_1 = x_test.iloc[:,5]
    # # print(type(ask_1),type(bid_1),type(y_predict))
    # utils.cal_average_pnl(y_predict,ask_1,bid_1)
    # print("模型测试完毕\n------------------------------------\n")    

# test_for_output_possibility(stock_id = 7,label_number=2)

# utils.cal_average_pnl([1,2,3],[2,3,4],[1,2,0],window=1)

# for depth in range(1,10,1):
#     params_default['max_depth'] = depth
#     logger.success(f"\nmax_depth:{depth}")
#     train_and_test(stock_id=7,label_number=2,params=params_default)    

# for weight in range(10,25,1):
#     weights = [1,weight/10,1]
#     logger.success(f"\nweights:{weights}")
#     train_and_test(stock_id=7,label_number=2,weights=weights)

# for num_found in range(10,110,10):
#     logger.success(f"\nnum_round:{num_found}")
#     train_and_test(stock_id=7,label_number=0,num_round=num_found)

# for gamma in np.arange(0.5, 0, -0.1):
#     params_default['gamma'] = gamma
#     logger.success(f"\ngamma:{gamma}")
#     train_and_test(stock_id=7,label_number=0,params=params_default)

# for stock_id in range(0,10):
#     train_and_test(stock_id=stock_id,label_number=2)
    
# train_and_test(stock_id=0)
# train_and_test(stock_id=1,label_number=0,train_ratio=1)

# for num_round in range(10,100,10):
#     train_and_test(stock_id=7,label_number=0,num_round=num_round)

def excute_for_all_stock_and_label():
    for stock_id in range(10):
        for label_number in range(5):
            train_and_test(stock_id = stock_id,label_number=label_number)    
       
# excute_for_all_stock_and_label()       
     
def search_for_depth():
    params = params_default.copy()
    for depth in range(1,5):
        params['max_depth'] = depth
        logger.success(f"\nmax_depth:{depth}")
        train_and_test(stock_id=1,label_number=0,params=params)
        
# search_for_depth()

def search_for_min_child_weight():
    params = params_default.copy()
    for min_child_weight in range(1,6):
        params['min_child_weight'] = min_child_weight
        logger.success(f"\nmin_child_weight:{min_child_weight}")
        train_and_test(stock_id=1,label_number=0,params=params)
        
# search_for_min_child_weight()

def search_for_sub_sample():
    params = params_default.copy()
    for sub_sample in [0.5,0.6,0.7,0.8,0.9,1]:
        params['subsample'] = sub_sample
        logger.success(f"\nsubsample:{sub_sample}")
        train_and_test(stock_id=1,label_number=0,params=params)
        
# search_for_sub_sample()

def search_for_num_round():
    params = params_default.copy()
    for num_round in range(50,200,25):
        logger.success(f"\nnum_round:{num_round}")
        train_and_test(stock_id=1,label_number=0,params=params,num_round=num_round)
        
# search_for_num_round()
        
        