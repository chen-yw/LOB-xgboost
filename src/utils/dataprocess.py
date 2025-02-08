from .logger import logger
import os
import numpy as np
import pandas as pd
from natsort import natsorted

import numpy as np
from multiprocessing import Pool

from utils import features as ft1
from utils import new_features as ft2

def data_process_FI(general_hyperparameters) -> None:
    """划分训练集，验证集，测试集，并保存"""
    logger.info("Processing the FI-2010 data.")
    # 获取当前文件的目录
    current_dir = os.path.dirname(__file__)
    # 获取当前文件的上一级目录
    parent_dir = os.path.dirname(current_dir)

    data_path = os.path.join(parent_dir, 'data', 'FI-2010')
    dec_data = np.loadtxt(os.path.join(data_path,'Train_Dst_NoAuction_DecPre_CF_7.txt'))
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * general_hyperparameters['training_ratio']))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * general_hyperparameters['validation_ratio'])):]
    # 保存训练集和验证集
    np.savetxt(os.path.join(data_path, 'train.txt'), dec_train,fmt='%.2e')
    np.savetxt(os.path.join(data_path, 'validation.txt'), dec_val,fmt='%.2e')
    dec_test1 = np.loadtxt(os.path.join(data_path,'Test_Dst_NoAuction_DecPre_CF_7.txt'))
    dec_test2 = np.loadtxt(os.path.join(data_path,'Test_Dst_NoAuction_DecPre_CF_8.txt'))
    dec_test3 = np.loadtxt(os.path.join(data_path,'Test_Dst_NoAuction_DecPre_CF_9.txt'))
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))


    # 保存测试集
    np.savetxt(os.path.join(data_path, 'test.txt'), dec_test,fmt='%.2e')
    logger.success("Data processing completed.")



def get_stock_FBDQA(stock)->pd.DataFrame:
    """获取FBDQA数据集中的某个股票的数据"""
    # 获取当前文件的目录
    current_dir = os.path.dirname(__file__)
    # 获取当前文件的上一级目录
    parent_dir = os.path.dirname(current_dir)

    data_path = os.path.join(parent_dir, 'data', 'FBDQA', 'data')
    # 数据格式为 snapshot_sym{id}_date{date}_{am/pm}.csv 遍历读取所有的csv文件，并合并
    data = pd.DataFrame()
    file_list = [file for file in os.listdir(data_path) if file.startswith(f'snapshot_sym{stock}')]

    # 按文件名中的日期和时间排序
    file_list= natsorted(file_list,key=lambda x: (x.split('_')[2], x.split('_')[3]))

    for file in file_list:
        file_path = os.path.join(data_path, file)
        data = pd.concat([data, pd.read_csv(file_path)], ignore_index=True)
    return data

def data_process_FBDQA(general_hyperparameters) -> None:
    """划分训练集，验证集，测试集，并保存"""
    logger.info("Processing the FBDQA dataset.")
    # 获取当前文件的目录
    current_dir = os.path.dirname(__file__)
    # 获取当前文件的上一级目录
    parent_dir = os.path.dirname(current_dir)

    data_path = os.path.join(parent_dir, 'data', 'FBDQA', 'data')

    # 根据training_stocks和target_stocks聚合与分割数据
    training_stocks = general_hyperparameters['training_stocks'].split(',')
    target_stocks = general_hyperparameters['target_stocks'].split(',')

    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for stock in training_stocks:
        data = get_stock_FBDQA(stock)
        if stock in target_stocks:
            train_data = pd.concat([train_data, data.iloc[:int(len(data)* general_hyperparameters['training_ratio'])]], ignore_index=True)
            val_data = pd.concat([val_data, data.iloc[int(len(data)* general_hyperparameters['training_ratio']):int(len(data)* (general_hyperparameters['training_ratio']+general_hyperparameters['validation_ratio']))]], ignore_index=True)
            test_data = pd.concat([test_data, data.iloc[int(len(data)* (general_hyperparameters['training_ratio']+general_hyperparameters['validation_ratio'])):]], ignore_index=True)
        else:
            train_data = pd.concat([train_data, data], ignore_index=True)
    for stock in target_stocks:
        data = get_stock_FBDQA(stock)
        if stock not in training_stocks:
            test_data = pd.concat([test_data, data], ignore_index=True)
    
    # 生成路径，如果不存在则创建
    temp_data_path = os.path.join(parent_dir,'data','FBDQA','temp_data')
    os.makedirs(temp_data_path, exist_ok=True)

    # 如果第18列的数字小于1e-10，则删除该行
    train_data = train_data[train_data.iloc[:,18] >= 1e-10]
    val_data = val_data[val_data.iloc[:,18] >= 1e-10]
    test_data = test_data[test_data.iloc[:,18] >= 1e-10]

    # 如果第8列的数字小于1e-10，则删除该行
    train_data = train_data[train_data.iloc[:,8] >= 1e-10]
    val_data = val_data[val_data.iloc[:,8] >= 1e-10]
    test_data = test_data[test_data.iloc[:,8] >= 1e-10]


    # 保存训练集和验证集
    train_data.to_csv(os.path.join(temp_data_path,'train.csv'), index=False)
    val_data.to_csv(os.path.join(temp_data_path,'validation.csv'), index=False)
    # 保存测试集
    test_data.to_csv(os.path.join(temp_data_path,'test.csv'), index=False)
    logger.success("Data processing completed.")

def extract_features(window_data):
    timestamp,bid, ask,bid_size,ask_size,price,volume =window_data
    features = []
    # 加上所有的基本特征
    features += bid[-1].tolist()
    features += ask[-1].tolist()
    features += bid_size[-1].tolist()
    features += ask_size[-1].tolist()
    features += [price[-1]]
    features += [volume[-1]]
    features += [timestamp[-1]]

    # todo 在这里加特征函数
    features += ft1.strength_pos_buy(bid, ask, volume, price)
    features += ft1.strength_pos_buy_version2(bid, ask, volume, price)
    features += ft1.norm_active_amount(volume, price)
    features += [ft1.t_active_amount(volume, price, d=10)]
    features += [ft1.t_active_amount(volume, price, d=20)]
    features += [ft1.s_big(bid, ask, volume, price)]
    features += [ft1.s_small(bid, ask, volume, price)]
    features += ft1.up_jump(bid,ask)
    features += ft1.dn_jump(bid,ask)
    features += ft1.net_jump(bid,ask)
    features += ft1.vol_vol(volume)
    features += ft1.jump_degree(price)
    features += [ft1.consequent_bid_ask_ratio(bid, ask)]
    features += ft1.non_fluid_factor(volume, price)
    features += ft1.consequent_bid(bid)
    features += ft1.consequent_ask(ask)
    features += ft1.up_ret_vol(ask,bid)
    features += ft1.dn_ret_vol(ask,bid)
    features += ft1.im_up_dn(bid,ask)
    features += ft1.ret_skew(bid,ask)
    features += ft1.RV(bid,ask)
    features += ft1.corr_v_r_rate(volume,price)
    features += ft1.acma(volume)
    features += ft1.ofi(bid,ask,bid_size,ask_size)
    features += ft1.MPC(bid,ask)
    features += [ft1.MPC_skew(bid,ask)]
    features += [ft1.Oimb(bid_size,ask_size)]
    features += [ft1.Timb(bid,price,volume)]
    features += ft1.Dimbi(bid_size,ask_size)
    features += ft1.Himbi(bid,ask)
    features += [ft1.volume_ave(volume)]
    features += [ft1.volume_std(volume)]
    features += [ft1.aggresive_bid_rate(bid,price,volume)]
    features += [ft1.aggresive_ask_rate(ask,price,volume)]
    features += [ft1.HL(price)]
    features += ft1.WAP(bid,ask,bid_size,ask_size)
    features += ft1.RSpread(bid,ask)
    features += ft1.WAPbidDiff(bid,ask,bid_size,ask_size)
    features += ft1.WAP_sq_rtn(bid,ask,bid_size,ask_size)
    features += ft1.Sspread(bid,ask)
    features += ft1.norm_ask_size(ask_size)
    features += [ft1.EXPratio(bid_size,ask_size)]

    features += ft2.quote_imbalance(bid,ask)
    features += [ft2.lit_trade_imbalance(price,ask)]
    features += ft2.average_liquid(bid_size,ask_size)
    features += ft2.liquid_std_variance(bid_size,ask_size)
    features += [ft2.liquid_skew(bid_size,ask_size)]
    features += [ft2.normalized_liquid(bid_size,ask_size)]
    features += [ft2.TED_1(bid_size,ask_size)]
    features += [ft2.TED_2(bid_size,ask_size)]
    features += [ft2.liquid_lack_delta_time(bid_size,ask_size,ft2.cal_liquid_lack(bid_size,ask_size))]
    features += [ft2.liquid_lack_cause(price,ft2.cal_liquid_lack(bid_size,ask_size))]
    features += [ft2.liquid_lack_average_price_change(price,ft2.cal_liquid_lack(bid_size,ask_size))]
    features += [ft2.bid_average_age(bid)]
    features += [ft2.ask_Average_age(ask)]
    features += [ft2.bid_change_count(bid)]
    features += [ft2.ask_change_count(ask)]
    features += ft2.bid_quantity_sensitive(bid,bid_size)
    features += ft2.ask_quantity_sensitive(ask,ask_size)
    features += [ft2.bid_ask_sensitive(bid,ask,bid_size,ask_size)]

    features += ft2.spread_all(bid,ask).tolist()
    features += ft2.midprice_all(bid,ask).tolist()
    features += ft2.relative_spread(bid,ask).tolist()
    features += [ft2.ask_mean(ask)]
    features += [ft2.bid_mean(bid)]
    features += [ft2.volume_diff(bid_size,ask_size)]
    features += [ft2.volume_diff_first_level(bid_size,ask_size)]
    features += [ft2.amount_size(bid_size,ask_size,volume)]
    features += [ft2.mid_price_range_5(bid,ask)]
    features += [ft2.mid_price_range_10(bid,ask)]
    features += [ft2.mid_price_range_20(bid,ask)]
    features += [ft2.bsize_asize_1(bid_size,ask_size)]
    features += [ft2.bsize_asize_3(bid_size,ask_size)]

    features += [ft2.rsi_price(price)]
    features += [ft2.rsi_volume(volume)]
    features += [ft2.rsi_bid(bid_size)]
    features += [ft2.rsi_ask(ask_size)]
    features += [ft2.average_sign_volume(volume)]
    # todo 在上面加特征函数
    
    return features

# 定义用于处理单个窗口的函数
def process_window(args):
    end_idx, data_numpy,labels, window_size = args

    start_idx = max(0, end_idx - window_size)  # 确保不超出边界
    # 获取窗口数据，窗口的结束位置是 end_idx，开始位置是 end_idx - window_size
    bid = data_numpy[start_idx:end_idx, :5]
    ask = data_numpy[start_idx:end_idx, 5:10]
    bid_size = data_numpy[start_idx:end_idx,10:15]
    ask_size = data_numpy[start_idx:end_idx,15:20]
    price = data_numpy[start_idx:end_idx,20]
    volume = data_numpy[start_idx:end_idx, 21]
    timestamp = data_numpy[start_idx:end_idx, 22]
    
    window_data = (timestamp,bid, ask,bid_size,ask_size,price,volume)

    features = extract_features(window_data) + labels.iloc[end_idx-1].tolist()



    return (end_idx , features)

# general_hyperparameters
def FBDQA_ML() -> None:
    """使用人工特征进行机器学习的数据预处理，数据保存在data/FBDQA/ML_data"""
    # 批量提取特征的函数


    def process_time_series_parallel(data, window_size=100, num_processes=4):
        # 丢弃前3列
        columns = ['n_bid1', 'n_bid2', 'n_bid3', 'n_bid4', 'n_bid5', 
                   'n_ask1', 'n_ask2', 'n_ask3', 'n_ask4', 'n_ask5', 
                   'n_bsize1', 'n_bsize2', 'n_bsize3', 'n_bsize4', 'n_bsize5', 
                   'n_asize1', 'n_asize2', 'n_asize3', 'n_asize4', 'n_asize5', 
                   'n_close', 'amount_delta', 'time']
        df = data[columns]
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.hour * 3600 + df['time'].dt.minute * 60 + df['time'].dt.second
        df['time'] = ((df['time'] - (9 * 3600 + 1800)) // 1800) + 1

        labels_col = ['label_5','label_10','label_20','label_40','label_60']
        labels = data[labels_col]
        # 将 pandas DataFrame 转换为 numpy 数组
        data_numpy = df.to_numpy()
        # 使用多进程处理
        with Pool(processes=num_processes) as pool:
            # range(0, len(df)) 代表从第一行开始处理到最后一行
            # 这里的 end_idx 指的是每个窗口的最后一行的索引
            results = pool.map(process_window, [(end_idx, data_numpy,labels, window_size) for end_idx in range(window_size, len(df) + 1)])

        logger.success('parallel complete')

        # 将结果按时间顺序排序
        results.sort(key=lambda x: x[0])  # 根据 end_idx 排序，保证时间顺序

        # 提取特征和标签
        sorted_results = [result[1] for result in results]  # 只取特征和标签部分
        
        # 将处理结果拼接成 DataFrame
        features_df = pd.DataFrame(sorted_results)

        logger.debug('features + labels shape:{}'.format(features_df.shape))
        return features_df

    # 假设你有一个原始数据 DataFrame df
    # 例如 df = pd.DataFrame({...})
    logger.info("Processing the FBDQA dataset for ML.")
    # 获取当前文件的目录
    current_dir = os.path.dirname(__file__)
    # 获取当前文件的上一级目录
    parent_dir = os.path.dirname(current_dir)

    stocks = [0,1,2,3,4,5,6,7,8,9]

    for stock in stocks:
        logger.info("Processing stock: {}".format(stock))
        df = get_stock_FBDQA(stock)
        # 使用方法：处理整个数据集
        # 获取可用核心数量
        num_processes = os.cpu_count()
        features_df = process_time_series_parallel(df, window_size=300,num_processes=num_processes-2)
        logger.success("Feature extraction completed for stock {}".format(stock))
        # 存储到文件中
        ml_data_path = os.path.join(parent_dir, 'data', 'FBDQA', 'ML_data_70')
        os.makedirs(ml_data_path, exist_ok=True)
        features_df.to_csv(os.path.join(ml_data_path, f'stock_{stock}_features.csv'), index=False)
        logger.success("Feature extraction and saving completed for stock {}".format(stock))

    logger.success("Feature extraction and saving completed for all stocks.")