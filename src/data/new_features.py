import numpy as np
from scipy.stats import norm
from scipy.stats import skew
from numba import jit

# 下面的数据是传入信息的示例
bid = np.array([[3,2,1],[4,3,2],[5,4,3]])
ask = np.array([[6,7,8],[7,8,9],[8,9,10]])
volume = np.array([2,1,3])
price = np.array([3,7,5])
bid_size = np.array([[1,2,3],[2,3,4],[3,4,5]])
ask_size = np.array([[2,1,3],[3,2,4],[4,3,5]])
'''
bid里面分别对应了每一时刻的买方各档价格
ask里面分别对应了每一时刻的卖方各档价格
volume里面是每一时刻的成交量（好像还没有归一化）
price是每一时刻的成交价格（已经实现了归一化）
bid_size是每一时刻的买方各档数量
ask_size是每一时刻的卖方各档数量
注意所有的时间截面都是越靠前，时间越靠前
'''

# 四篇论文里面的特征

'''
Behind Stock Price Movement: Supply and Demand in Market Microstructure and Market Influence
Daily Volume Forecasting using High Frequency Predictors
Designating market maker behaviour in Limit Order Book markets
Effects of the Limit Order Book on Price Dynamics
'''

def extract_window(data, begin, end):
    '''提取数据窗口'''
    # 初始化一个空列表来存储提取的数据窗口
    window = []
    # 遍历元组中的每个数据序列
    for sequence in data:
        # 提取从 begin 到 end 的数据窗口，并添加到 window 列表中
        window.append(sequence[begin:end])
    return tuple(window)  # 将列表转换为元组并返回


def _quote_imbalance(bid, ask):
    '''报价不平衡，反应最佳买单量和最佳卖单量的变化'''
    delta_q_b = bid[-1,0] - bid[-2,0]
    delta_q_a = ask[-1,0] - ask[-2,0]
    return delta_q_b - delta_q_a

def quote_imbalance(bid, ask):
    '''报价不平衡，反应最佳买单量和最佳卖单量的变化'''
    data = (bid, ask)
    data_1 = extract_window(data, -10, -5)
    data_2 = extract_window(data, -20, -10)
    r1 = _quote_imbalance(*data_1)
    r2 = _quote_imbalance(*data_2)
    return [r1, r2]

def lit_trade_imbalance(price,ask):
    '''买方驱动和卖方驱动的指标。如果价格高于上一时刻最低卖价，表明买方主导，反之类似'''
    return price[-1] - ask[-2,0]

# -------------------一系列有关流动性的因子-------------------

def _average_liquid(bid_size,ask_size):
    '''平均流动性'''
    total_liquid = bid_size + ask_size
    sum_liquid = np.sum(total_liquid,axis=1)
    return np.mean(sum_liquid)

def average_liquid(bid_size,ask_size):
    '''平均流动性'''
    data = (bid_size, ask_size)
    data_1 = extract_window(data, -10, -5)
    data_2 = extract_window(data, -20, -10)
    r0 = _average_liquid(*data)
    r1 = _average_liquid(*data_1)
    r2 = _average_liquid(*data_2)
    return [r0, r1, r2]
    
def _liquid_std_variance(bid_size,ask_size):
    '''流动性的标准差'''
    total_liquid = bid_size + ask_size
    sum_liquid = np.sum(total_liquid,axis=1)
    return np.std(sum_liquid)

def liquid_std_variance(bid_size,ask_size):
    '''流动性的标准差'''
    data = (bid_size, ask_size)
    data_1 = extract_window(data, -10, -5)
    data_2 = extract_window(data, -20, -10)
    r0 = _liquid_std_variance(*data)
    r1 = _liquid_std_variance(*data_1)
    r2 = _liquid_std_variance(*data_2)
    return [r0, r1, r2]
    
def liquid_skew(bid_size,ask_size):
    '''流动性的偏度'''
    total_liquid = bid_size + ask_size
    sum_liquid = np.sum(total_liquid,axis=1)
    return skew(sum_liquid)

def normalized_liquid(bid_size,ask_size):
    '''归一化流动性，用来反应此时订单薄的流动性相对于历史的位置'''
    total_liquid = bid_size + ask_size
    sum_liquid = np.sum(total_liquid,axis=1)
    liquid_mean = np.mean(sum_liquid)
    liquid_std = np.std(sum_liquid)
    # print(sum_liquid)
    # print(liquid_mean)
    # print(liquid_std)
    return (sum_liquid[-1] - liquid_mean) / liquid_std


def TED_1(bid_size,ask_size):
    '''描述流动性的韧劲，低于平均值一个标准差之后恢复的平均用时'''
    total_liquid = bid_size + ask_size
    sum_liquid = np.sum(total_liquid,axis=1)
    # print(sum_liquid)
    liquid_mean = np.mean(sum_liquid)
    liquid_std = np.std(sum_liquid)
    liquid_lack = sum_liquid < liquid_mean - liquid_std
    liquid_lack.astype(int)
    # print(liquid_lack)
    # 现在将liquid_lack转化为一个序列，1代表缺乏流动性，0代表正常
    sum = np.sum(liquid_lack)
    diff = np.diff(liquid_lack)
    count = 0
    for i in liquid_lack:
        if i > 0:
            count += 1
    return sum / (count + 1)
    

def TED_2(bid_size,ask_size):
    '''描述流动性的韧劲，低于平均值两个标准差之后恢复的平均用时'''
    total_liquid = bid_size + ask_size
    sum_liquid = np.sum(total_liquid,axis=1)
    # print(sum_liquid)
    liquid_mean = np.mean(sum_liquid)
    liquid_std = np.std(sum_liquid)
    liquid_lack = sum_liquid < liquid_mean - 2*liquid_std
    liquid_lack.astype(int)
    # print(liquid_lack)
    # 现在将liquid_lack转化为一个序列，1代表严重缺乏流动性，0代表正常
    sum = np.sum(liquid_lack)
    diff = np.diff(liquid_lack)
    count = 0
    for i in liquid_lack:
        if i > 0:
            count += 1
    return sum / (count + 1)

def cal_liquid_lack(bid_size,ask_size):
    '''中间指标，用来判断是否出现了流动性匮乏的现象（请不要在训练中使用）'''
    total_liquid = bid_size + ask_size
    sum_liquid = np.sum(total_liquid,axis=1)
    liquid_mean = np.mean(sum_liquid)
    liquid_std = np.std(sum_liquid)
    liquid_lack = sum_liquid < liquid_mean - liquid_std
    liquid_lack.astype(int)
    return liquid_lack

def liquid_lack_delta_time(bid_size,ask_size,liquid_lack):
    '''计算上一次发生流动性缺乏到现在的时间间隔'''
    '''需要注意的是，这个函数需要用cal_liquid_lack函数得到的liquid_lack作为输入'''
    for i in range(1,len(liquid_lack)):
        if liquid_lack[-i] == 1:
            return i-1
    return len(liquid_lack)

def liquid_lack_cause(price,liquid_lack):
    '''如果最后一次出现了流动性匮乏现象，判断流动性缺乏与上涨还是下跌关联'''
    '''需要注意的是，这个函数需要用cal_liquid_lack函数得到的liquid_lack作为输入'''
    if liquid_lack[-1] == 1:
        if price[-1] > price[-2]:
            return 1
        else:
            return -1
    else:
        return 0
    
def liquid_lack_average_price_change(price,liquid_lack):
    '''判断一段时间当中，出现流动性缺乏的时候，价格是上涨还是下跌。如果大多上涨，那么说明多头力量强大'''
    '''需要注意的是，这个函数需要用cal_liquid_lack函数得到的liquid_lack作为输入'''
    delta_price = np.diff(price)
    # print(delta_price)
    for i in range(0, len(delta_price)):
        delta_price[i] = delta_price[i] * liquid_lack[i+1]
    return np.mean(delta_price)    
    
    
def bid_average_age(bid):
    '''买单的平均年龄'''
    now_bid = bid[-1,:]
    age_sum = 0 # 用来记录所有买价的年龄之和
    bid_count = 0 # 用来记录一共有几个买价
    for every_bid in now_bid:
        bid_count += 1 
        for i in range(-2,-len(bid)-1,-1):
            if every_bid in bid[i,:]:
                age_sum += 1
            else:
                break
    return age_sum/bid_count    
    
def ask_Average_age(ask):
    '''卖单的平均年龄'''
    now_ask = ask[-1,:]
    age_sum = 0 # 用来记录所有卖价的年龄之和
    ask_count = 0 # 用来记录一共有几个卖价
    for every_ask in now_ask:
        ask_count += 1 
        for i in range(-2,-len(ask)-1,-1):
            if every_ask in ask[i,:]:
                age_sum += 1
            else:
                break
    return age_sum/ask_count
    
def bid_change_count(bid):
    '''这一时刻相对于上一时刻有多少个买价发生了变化'''
    now_bid = bid[-1,:]
    count = len(now_bid)
    for every_bid in now_bid:
        if every_bid in bid[-2,:]:
            count -= 1
    return count    
    
def ask_change_count(ask):
    '''这一时刻相对于上一时刻有多少个卖价发生了变化'''
    now_ask = ask[-1,:]
    count = len(now_ask)
    for every_ask in now_ask:
        if every_ask in ask[-2,:]:
            count -= 1
    return count

def _bid_quantity_sensitive(bid,bid_size):
    '''买价的数量敏感度'''
    now_bid = bid[-1,:]
    now_bid_size = bid_size[-1,:]
    return (now_bid[0] - now_bid[len(now_bid)-1]) / np.sum(now_bid_size)

def bid_quantity_sensitive(bid,bid_size):
    data = (bid, bid_size)
    data_1 = extract_window(data, -10, -5)
    data_2 = extract_window(data, -20, -10)
    r1 = _bid_quantity_sensitive(*data_1)
    r2 = _bid_quantity_sensitive(*data_2)
    return [r1, r2]

def _ask_quantity_sensitive(ask,ask_size):
    '''卖价的数量敏感度'''
    now_ask = ask[-1,:]
    now_ask_size = ask_size[-1,:]
    return (now_ask[len(now_ask)-1] - now_ask[0]) / np.sum(now_ask_size)

def ask_quantity_sensitive(ask,ask_size):
    data = (ask, ask_size)
    data_1 = extract_window(data, -10, -5)
    data_2 = extract_window(data, -20, -10)
    r1 = _ask_quantity_sensitive(*data_1)
    r2 = _ask_quantity_sensitive(*data_2)
    return [r1, r2]

def bid_ask_sensitive(bid,ask,bid_size,ask_size):
    '''买卖敏感度程度之比'''
    bid_sen = _bid_quantity_sensitive(bid,bid_size)
    ask_sen = _ask_quantity_sensitive(ask,ask_size)
    return bid_sen/ask_sen

'''
下面是根据往届报告提取出来的因子
'''
def spread_all(bid,ask):
    '''买卖价之间的差距，返回的是一个列表'''
    return ask[-1,:] - bid[-1,:]

def spread_1(bid,ask):
    '''买一卖一价之间的差距'''
    return ask[-1,0] - bid[-1,0]

def spread_2(bid,ask):
    '''买二卖二价之间的差距'''
    return ask[-1,1] - bid[-1,1]

def spread_3(bid,ask):
    '''买三卖三价之间的差距'''
    return ask[-1,2] - bid[-1,2]

def spread_4(bid,ask):
    '''买四卖四价之间的差距'''
    return ask[-1,3] - bid[-1,3]

def spread_5(bid,ask):
    '''买五卖五价之间的差距'''
    return ask[-1,4] - bid[-1,4]

def midprice_all(bid,ask):
    '''所有档位的中间价'''
    return (ask[-1,:] + bid[-1,:]) / 2

def midprice_1(bid,ask):
    '''一档中间价'''
    return (ask[-1,0] + bid[-1,0]) / 2

def midprice_2(bid,ask):
    '''二档中间价'''
    return (ask[-1,1] + bid[-1,1]) / 2

def midprice_3(bid,ask):
    '''三档中间价'''
    return (ask[-1,2] + bid[-1,2]) / 2

def midprice_4(bid,ask):
    '''四档中间价'''
    return (ask[-1,3] + bid[-1,3]) / 2

def midprice_5(bid,ask):
    '''五档中间价'''
    return (ask[-1,4] + bid[-1,4]) / 2

def relative_spread(bid,ask):
    '''价差和均价的比值，返回的是一个列表'''
    spread = spread_all(bid,ask)
    midprice = midprice_all(bid,ask)
    return spread / midprice

def ask_mean(ask):
    '''卖一价的均值'''
    return np.mean(ask[:,0])

def bid_mean(bid):
    '''买一价的均值'''
    return np.mean(bid[:,0])

def volume_diff(bid_size,ask_size):
    '''买卖量之差'''
    now_bid = bid_size[-1,:]
    now_ask = ask_size[-1,:]
    return np.sum(now_bid) - np.sum(now_ask)

def volume_diff_first_level(bid_size,ask_size):
    '''一档买卖量之差'''
    return bid_size[-1,0] - ask_size[-1,0]

def amount_size(bid_size,ask_size,volume):
    '''交易量与买卖报价单量结合'''
    return volume[-1] / (bid_size[-1,0] + ask_size[-1,0])

def mid_price_range_5(bid,ask):
    '''中间价变化趋势（与5个tick之前作差）'''
    return (ask[-1,0] + bid[-1,0]) / 2 - (ask[-6,0] + bid[-6,0]) / 2

def mid_price_range_10(bid,ask):
    '''中间价变化趋势（与10个tick之前作差）'''
    return (ask[-1,0] + bid[-1,0]) / 2 - (ask[-11,0] + bid[-11,0]) / 2

def mid_price_range_20(bid,ask):
    '''中间价变化趋势（与20个tick之前作差）'''
    return (ask[-1,0] + bid[-1,0]) / 2 - (ask[-21,0] + bid[-21,0]) / 2

def bsize_asize_1(bid_size,ask_size):
    '''一档报单与总报单量之比'''
    now_bid = bid_size[-1,:]
    now_ask = ask_size[-1,:]
    return (now_bid[0] + now_ask[0]) / (np.sum(now_bid) + np.sum(now_ask))

def bsize_asize_3(bid_size,ask_size):
    '''三档报单与总报单量之比'''
    now_bid = bid_size[-1,:]
    now_ask = ask_size[-1,:]
    return (np.sum(now_bid[:3]) + np.sum(now_ask[:3])) / (np.sum(now_bid) + np.sum(now_ask))

'''
下面是根据SRT提取出来的因子
'''
def rsi_price(price):
    '''价格的RSI'''
    delta = np.diff(price)
    up = delta[delta>0].sum()
    down = -delta[delta<0].sum()
    return up / (up + down)

def rsi_volume(volume):
    '''成交量的RSI'''
    delta = np.diff(volume)
    up = delta[delta>0].sum()
    down = -delta[delta<0].sum()
    return up / (up + down)

def rsi_bid(bid_size):
    '''买单量的RSI'''
    bid_size = np.sum(bid_size,axis=1)
    delta = np.diff(bid_size)
    up = delta[delta>0].sum()
    down = -delta[delta<0].sum()
    return up / (up + down)

def rsi_ask(ask_size):
    '''卖单量的RSI'''
    ask_size = np.sum(ask_size,axis=1)
    delta = np.diff(ask_size)
    up = delta[delta>0].sum()
    down = -delta[delta<0].sum()
    return up / (up + down)

def average_sign_volume(volume):
    '''放量比例'''
    average_volume = np.mean(volume)
    std_var_colume = np.std(volume)
    sign_volume = [volume>average_volume+std_var_colume]
    return np.mean(sign_volume)

