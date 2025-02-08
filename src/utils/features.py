# 负责计算订单簿上的因子

'''
- 订单簿中含有的信息类型可能包括
    - 买方各档的价格 bid
    - 买方各档的数量 bid_size
    - 卖方各档的价格 ask
    - 卖方各档的数量 ask_size
    - 最新成交价 price
    - 最新成交量 volume
    - 日内时间戳 timestamp
- 订单簿的数据类型统一使用np.array，其中行代表不同的时间戳（越近的时间戳越靠后），列代表不同的价格档位（最佳价格在最左端）
'''
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
# from tick_pool.py

def extract_window(data, begin, end):
    '''提取数据窗口'''
    # 初始化一个空列表来存储提取的数据窗口
    window = []
    # 遍历元组中的每个数据序列
    for sequence in data:
        # 提取从 begin 到 end 的数据窗口，并添加到 window 列表中
        window.append(sequence[begin:end])
    return tuple(window)  # 将列表转换为元组并返回

def _strength_pos_buy(bid,ask,volume,price):
    '''主买主卖力量强弱对比'''
    vwap = np.sum(volume * price) / np.sum(volume)
    midp = (ask[-1,0] + bid[-1,0]) / 2
    r = (vwap - midp) * 2 / (ask[-1,0] - bid[-1,0])
    return r

def strength_pos_buy(bid, ask, volume, price):
    data = (bid, ask, volume, price)
    data_1 = extract_window(data, -20, None)
    data_2 = extract_window(data, -40, -20)
    r1 = _strength_pos_buy(*data_1)
    r2 = _strength_pos_buy(*data_2)
    return [r1, r2]

def _strength_pos_buy_version2(bid, ask, volume, price):
    '''第二种定义主买的方法,计算本窗口中主买的次数占主买主卖次数和的比例'''
    # Calculate VWAP (volume weighted average price) at time t
    vwap = np.sum(volume * price) / np.sum(volume)
    # Calculate mid-price  (average of best bid and best ask)
    midp = (ask[:, 0] + bid[:, 0]) / 2
    # Generate buy (b) and sell (s) signals based on the comparison
    b = vwap > midp
    s = vwap < midp
    # Calculate the sum
    b_sum = np.sum(b.astype(int))
    s_sum = np.sum(s.astype(int))
    # Calculate the ratio of buy signals over the sum of buy and sell signals
    f = b_sum / (b_sum + s_sum)
    # Return the result, padding with zeros to maintain the array size
    return f

def strength_pos_buy_version2(bid, ask, volume, price):
    '''主买主卖力量强弱对比'''
    data = (bid, ask, volume, price)
    data_1 = extract_window(data, -20, None)
    data_2 = extract_window(data, -40, -20)
    r1 = _strength_pos_buy_version2(*data_1)
    r2 = _strength_pos_buy_version2(*data_2)
    return [r1, r2]


def _norm_active_amount(volume, price):
    '''置信正态分布主动占比，tick涨幅越大，主动占比越大。窗口内极限涨跌幅由均值和标准差决定'''
    r = price/np.roll(price,1)-1
    r = np.nan_to_num(r,nan=0)
    fiducial_amount = volume * norm.cdf(r / np.std(r) * 1.96)
    return np.sum(fiducial_amount) / np.sum(volume)

def norm_active_amount(volume, price):
    '''置信正态分布主动占比'''
    data = (volume, price)
    data_1 = extract_window(data, -40, None)
    data_2 = extract_window(data, -80, -40)
    r1 = _norm_active_amount(*data_1)
    r2 = _norm_active_amount(*data_2)
    return [r1, r2]

def t_active_amount(volume, price, d = 10):
    '''t分布主动占比 '''
    r = price/np.roll(price,1)-1
    r = np.nan_to_num(r,nan=0)
    fiducial_amount = volume * stats.t.cdf(r, d)
    return np.sum(fiducial_amount) / np.sum(volume)


def s_big(bid, ask, volume, price):
    '''来自于残差资金流强度因子构建，分子为大单买额-大单卖额,分母为abs(大单买额-大单卖额)。推荐数据窗口大于100'''
    # 计算 VWAP
    vwap = np.sum(volume * price) / np.sum(volume)
    # 计算中间价（中间价为买一卖一价的平均）
    midp = (ask[:, 0] + bid[:, 0]) / 2
    # 定义主买主卖
    b = vwap > midp  # 主买
    s = vwap < midp  # 主卖
    # 定义大单买卖条件
    big_order = volume > (np.mean(volume) + np.std(volume))  # 这里以成交量的滚动均值加标准差来替代原条件
    # 计算大单买额和卖额
    ba = (b * big_order) * volume
    sa = (s * big_order) * volume
    # 计算资金流强度因子
    numerator = np.sum(ba - sa)
    denominator = np.abs(np.sum(ba - sa))
    # 处理分母为0的情况
    if denominator == 0:
        return 0
    return numerator / denominator


def s_small(bid, ask, volume, price):
    '''来自于残差资金流强度因子构建，分子为小单买额-小单卖额，分母为abs(小单买额-小单卖额)'''
    # 计算 VWAP
    vwap = np.sum(volume * price) / np.sum(volume)
    # 计算中间价（中间价为买一卖一价的平均）
    midp = (ask[:, 0] + bid[:, 0]) / 2
    # 定义主买主卖
    b = vwap > midp  # 主买
    s = vwap < midp  # 主卖
    # 定义小单买卖条件
    small_order = volume < (np.mean(volume) - np.std(volume))  # 这里以成交量的滚动均值减标准差来替代原条件
    # 计算小单买额和卖额
    ba = (b * small_order) * volume
    sa = (s * small_order) * volume
    # 计算资金流强度因子
    numerator = np.sum(ba - sa)
    denominator = np.abs(np.sum(ba - sa))
    # 处理分母为0的情况
    if denominator == 0:
        return 0
    return numerator / denominator


# -------------------------------2 价格跳档--------------------------

def _up_jump(bid, ask):
    '''价格上涨，跳档次数，推荐数据窗口大于20'''
    # 获取买一价和前一时刻卖一价
    j = bid[:, 0] > np.roll(ask[:, 0], 1)
    return np.sum(j) / len(j)

def up_jump(bid, ask):
    '''价格上涨，跳档次数'''
    data = (bid, ask)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -50, -30)
    r1 = _up_jump(*data_1)
    r2 = _up_jump(*data_2)
    return [r1, r2]

def _dn_jump(bid, ask):
    '''价格下跌，跳档次数，推荐数据窗口大于20'''
    # 获取卖一价和前一时刻买一价
    j = ask[:, 0] < np.roll(bid[:, 0], 1)
    return np.sum(j) / len(j)

def dn_jump(bid, ask):
    '''价格下跌，跳档次数'''
    data = (bid, ask)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -50, -30)
    r1 = _dn_jump(*data_1)
    r2 = _dn_jump(*data_2)
    return [r1, r2]


def _net_jump(bid, ask):
    '''净跳档次数'''
    j1 = bid[:, 0] > np.roll(ask[:, 0], 1)  # 当前买一价大于前一时刻卖一价
    j2 = ask[:, 0] < np.roll(bid[:, 0], 1)  # 当前卖一价小于前一时刻买一价
    return (np.sum(j1) - np.sum(j2)) / len(j1)

def net_jump(bid, ask):
    '''净跳档次数'''
    data = (bid, ask)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -50, -30)
    r1 = _net_jump(*data_1)
    r2 = _net_jump(*data_2)
    return [r1, r2]

def _vol_vol(volume):
    '''高频成交量波动'''
    return np.std(volume) / np.mean(volume)

def vol_vol(volume):
    '''高频成交量波动'''
    data = (volume,)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -50, -30)
    r1 = _vol_vol(*data_1)
    r2 = _vol_vol(*data_2)
    return [r1, r2]

def _jump_degree(price):
    '''跳跃度，超参数为20'''
    rtn = price / np.roll(price, 1)
    ln_rtn = np.log(price / np.roll(price, 20))
    return ((rtn - ln_rtn) * 2 - ln_rtn ** 2)[-1]

def jump_degree(price):
    '''跳跃度'''
    data = (price,)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _jump_degree(*data_1)
    r2 = _jump_degree(*data_2)
    return [r1, r2]
#-------------------------------4 流动性类---------------------------

def consequent_bid_ask_ratio(bid, ask):
    '''滚动窗口中有多少个bid和ask是连续一样的?计算他们的差值'''
    same_ask = np.sum(ask[:, 0] == np.roll(ask[:, 0], 5))
    same_bid = np.sum(bid[:, 0] == np.roll(bid[:, 0], 5))
    diff = same_ask - same_bid
    return diff

def _non_fluid_factor(volume, price):
    '''经典非流动性因子,单位收益率由多少成交量推动'''
    epsilon = 1e-6
    ret = np.log(price / np.roll(price, 1))  # 对数收益率
    v = np.sum(volume) + 1e-6
    f = (np.sum(ret) / v ) * 1e10  # 乘以1e10确保在绝大多数情况下值可以被接受
    return f  # 单位收益率由多少成交量推动？

def non_fluid_factor(volume, price):
    '''经典非流动性因子'''
    data = (volume, price)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _non_fluid_factor(*data_1)
    r2 = _non_fluid_factor(*data_2)
    return [r1, r2]

def _consequent_bid(bid):
    '''滚动窗口中,有多少个bp1是一样的'''
    same_bid = bid[:, 0] == np.roll(bid[:, 0], 1)
    return np.sum(same_bid)

def consequent_bid(bid):
    '''滚动窗口中,有多少个bp1是一样的'''
    data = (bid,)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _consequent_bid(*data_1)
    r2 = _consequent_bid(*data_2)
    return [r1, r2]


def _consequent_ask(ask):
    '''滚动窗口中,多少个ap1是一样的?'''
    same_ask = ask[:, 0] == np.roll(ask[:, 0], 1)
    return np.sum(same_ask)

def consequent_ask(ask):
    '''滚动窗口中,多少个ap1是一样的?'''
    data = (ask,)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _consequent_ask(*data_1)
    r2 = _consequent_ask(*data_2)
    return [r1, r2]

# todo 没看懂
# def consumption_rates(volume, ask, bid): 
#     '''订单薄消耗速率'''
#     vol_d = volume
#     to_d = ask[:, 0] - bid[:, 0]
#     return np.sum(vol_d) / np.sum(to_d)

# -------------------------------5 收益率相关----------------------------------

def _up_ret_vol(ask,bid):
    '''高频上行波动占比'''
    # 计算中间价
    mid = (ask[:, 0] + bid[:, 0]) / 2
    # 计算对数收益并放大
    yRtn = np.diff(np.log(mid)) * 1e4
    # 计算上涨部分的平方和
    a = np.square(yRtn[yRtn > 0])
    # 计算所有波动的平方和
    b = np.square(yRtn)

    return np.sum(a) / np.sum(b)

def up_ret_vol(ask,bid):
    '''高频上行波动占比'''
    data = (ask, bid)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _up_ret_vol(*data_1)
    r2 = _up_ret_vol(*data_2)
    return [r1, r2]

def _dn_ret_vol(ask,bid):
    '''高频下行波动占比'''
    # 计算中间价
    mid = (ask[:, 0] + bid[:, 0]) / 2
    # 计算对数收益并放大
    yRtn = np.diff(np.log(mid)) * 1e4
    # 计算下跌部分的平方和
    a = np.square(yRtn[yRtn < 0])
    # 计算所有波动的平方和
    b = np.square(yRtn)

    return np.sum(a) / np.sum(b)

def dn_ret_vol(ask,bid):
    '''高频下行波动占比'''
    data = (ask, bid)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _dn_ret_vol(*data_1)
    r2 = _dn_ret_vol(*data_2)
    return [r1, r2]


def _im_up_dn(bid, ask):
    '''上下行波动率跳跃的不对称性'''
    # 计算中间价并向前移位
    mid = (ask[:, 0] + bid[:, 0]) / 2
    mid_shifted = np.roll(mid, -1)  # 将 mid 向前移动一个位置

    # 计算对数收益并限制在3倍标准差之内
    log_returns = np.log(mid_shifted) - np.log(mid)
    std_log_returns = np.std(log_returns)
    yRtn = np.clip(log_returns, -3 * std_log_returns, 3 * std_log_returns) * 1e4

    # 计算上涨部分的平方和
    u = np.sum(np.square(yRtn[yRtn > 0]))
    # 计算下跌部分的平方和
    d = np.sum(np.square(yRtn[yRtn < 0]))
    # 返回上下行波动的差异
    return u - d

def im_up_dn(bid, ask):
    '''上下行波动率跳跃的不对称性'''
    data = (bid, ask)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _im_up_dn(*data_1)
    r2 = _im_up_dn(*data_2)
    return [r1, r2]

def _ret_skew(bid, ask):
    '''高频已实现偏度，刻画股票日内快速拉升或下跌的特征，与收益率负相关'''
    # 计算中间价
    mid = (ask[:, 0] + bid[:, 0]) / 2
    # 计算对数收益的平方
    yRtn = np.square(np.diff(np.log(mid)))
    # 计算所有值的偏度
    n = len(yRtn)
    mean_yRtn = np.mean(yRtn)
    std_yRtn = np.std(yRtn)
    # 计算偏度
    skewness = (n / ((n - 1) * (n - 2))) * np.sum(((yRtn - mean_yRtn) ** 3) / (std_yRtn ** 3))
    return skewness

def ret_skew(bid, ask):
    '''高频已实现偏度'''
    data = (bid, ask)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _ret_skew(*data_1)
    r2 = _ret_skew(*data_2)
    return [r1, r2]

# -------------------------------6 量价相关性----------------------------------

def _RV(bid, ask):
    '''已实现波动率, 采用Kaggle的波动率计算方式'''
    # 计算中间价
    mid = (bid[:, 0] + ask[:, 0]) / 2
    # 计算对数收益率
    yRtn = np.log(mid,np.roll(mid,1))  # 对数收益率, 计算相邻两天的比值差
    # 计算过去n个时间点的已实现波动率
    rv = np.sqrt(np.sum(np.square(yRtn)))
    return rv

def RV(bid, ask):
    '''已实现波动率'''
    data = (bid, ask)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _RV(*data_1)
    r2 = _RV(*data_2)
    return [r1, r2]

def _corr_v_r_rate(volume, price):
    '''成交量变化率和收益率的相关性''' 
    # 计算收益率
    rtn = np.log(price / np.roll(price, 1))
    # 计算成交量变化率
    vol_rtn = volume / np.roll(volume, 1)
    # 计算相关性
    corr = np.corrcoef(rtn, vol_rtn)
    return corr[0, 1]

def corr_v_r_rate(volume, price):
    '''成交量变化率和收益率的相关性'''
    data = (volume, price)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _corr_v_r_rate(*data_1)
    r2 = _corr_v_r_rate(*data_2)
    return [r1, r2]

def _acma(volume):
    '''成交额自相关性'''
    # 计算成交额的自相关性
    ac = np.mean(np.correlate(volume, volume, mode='full'))
    return ac

def acma(volume):
    '''成交额自相关性'''
    data = (volume,)
    data_1 = extract_window(data, -30, None)
    data_2 = extract_window(data, -60, -30)
    r1 = _acma(*data_1)
    r2 = _acma(*data_2)
    return [r1, r2]

# -------------------------------7 订单薄失衡----------------------------------

def _ofi(bid, ask, bid_size, ask_size):
    '''订单薄失衡'''
    # 计算bid一侧
    bid_p_previous = np.roll(bid[:, 0], 1)
    bid_p_current = bid[:, 0]
    delta_v1 = (bid_p_current > bid_p_previous) * bid_size[:, 0]
    delta_v2 = (bid_p_current < bid_p_previous) * np.roll(bid_size[:, 0], 1) * -1
    delta_v3 = (bid_p_current == bid_p_previous) * (bid_size[:, 0] - np.roll(bid_size[:, 0], 1))
    delta_bid_v = delta_v1 + delta_v2 + delta_v3

    # 计算ask一侧
    ask_p_previous = np.roll(ask[:, 0], 1)
    ask_p_current = ask[:, 0]
    delta_v1 = (ask_p_current > ask_p_previous) * np.roll(ask_size[:, 0], 1) * -1
    delta_v2 = (ask_p_current < ask_p_previous) * ask_size[:, 0]
    delta_v3 = (ask_p_current == ask_p_previous) * (ask_size[:, 0] - np.roll(ask_size[:, 0], 1))
    delta_ask_v = delta_v1 + delta_v2 + delta_v3

    iof = delta_bid_v - delta_ask_v

    return np.sum(iof)

def ofi(bid, ask, bid_size, ask_size):
    '''订单薄失衡'''
    data = (bid, ask, bid_size, ask_size)
    data_1 = extract_window(data, -10, None)
    data_2 = extract_window(data, -40, -10)
    r1 = _ofi(*data_1)
    r2 = _ofi(*data_2)
    return [r1, r2]

# -------------------------------8 中间价变化率-------------------------------
def _MPC(bid, ask):
    '''中间价变化率'''
    # 计算中间价
    mid = (ask[:, 0] + bid[:, 0]) / 2
    # 计算中间价的变化率
    mid_rtn = mid / np.roll(mid, 1) - 1
    return np.sum(mid_rtn)

def MPC(bid, ask):
    '''中间价变化率'''
    data = (bid, ask)
    data_1 = extract_window(data, -10, None)
    data_2 = extract_window(data, -40, -10)
    r1 = _MPC(*data_1)
    r2 = _MPC(*data_2)
    return [r1, r2]

def MPC_skew(bid, ask):
    '''中间价变化率的偏度'''
    # 计算中间价
    mid = (ask[:, 0] + bid[:, 0]) / 2
    # 计算中间价的变化率
    mid_rtn = mid / np.roll(mid, 1) - 1
    # 计算偏度
    skew = np.mean(mid_rtn) / np.std(mid_rtn)
    return skew

# Identifying Expensive Trades by Monitoring the Limit Order Book

def Oimb(bid_size, ask_size):
    '''订单簿不平衡'''
    # 计算订单簿不平衡
    oimb = np.sum(bid_size[-15:, 0] - ask_size[-15:, 0]) / np.sum(bid_size[-15:, 0] + ask_size[-15:, 0])
    return oimb

def Timb(bid,price,volume):
    '''交易不平衡'''
    # 计算买一小于等于price的位置
    b = bid[:, 0] <= price
    b = b[-15:]
    volume = volume[-15:]
    # 计算主卖量：买一等于price的位置的成交量
    s = np.sum(volume[b])
    SUM = np.sum(volume)
    # 计算交易不平衡
    timb =  (SUM - s) - s/ SUM
    return timb

def Dimbi(bid_size, ask_size):
    '''深度不平衡'''
    dimbi = (bid_size[-1] - ask_size[-1])/(bid_size[-1] + ask_size[-1])
    dimbi = dimbi.tolist()
    return dimbi

def Himbi(bid,ask):
    '''价格高度不平衡'''
    # 计算同侧各档位的价格差
    bid = bid[-1]
    ask = ask[-1]
    diff_bid = np.diff(bid)
    diff_ask = np.diff(ask)
    # 计算价格高度不平衡
    himbi = (diff_ask - diff_bid) / (diff_ask + diff_bid)
    return himbi.tolist()

# todo 撤单因子，无法完成

# todo 市值和N天交易量因子，无法完成

def volume_ave(volume):
    '''成交量均值'''
    return np.mean(volume)

def volume_std(volume):
    '''成交量标准差'''
    return np.std(volume)

def aggresive_bid_rate(bid,price,volume):
    '''超越最佳买价的买方成交量占比'''
    b = bid[:,0] < price
    b = b[-20:]
    volume = volume[-20:]
    s = np.sum(volume[b])
    SUM = np.sum(volume)
    return s/SUM

def aggresive_ask_rate(ask,price,volume):
    '''超越最佳卖价的卖方成交量占比'''
    a = ask[:,0] > price
    a = a[-20:]
    volume = volume[-20:]
    s = np.sum(volume[a])
    SUM = np.sum(volume)
    return s/SUM

# def R_spread(bid,ask): # ! 弃用，见下同名函数
#     '''相对扩散率'''
#     return (ask[-1,0] - bid[-1,0]) / (ask[-1,0] + bid[-1,0]) * 2

def HL(price):
    '''高低价差'''
    return (np.max(price) - np.min(price))/np.mean(price)

# def Hidden_vol(bid_size,ask_size):
#     '''隐藏流动性'''
#     pass
# todo 没看懂

# Multivariate Realized Volatility Forecasting with Graph Neural Network 附录中的特征

def _gini(x):
    '''计算基尼系数，作为算子被调用'''
    # 计算排序后的累计和
    cumsum = np.cumsum(np.sort(x))
    # 计算基尼系数
    gini = np.sum((2 * np.arange(1, len(x) + 1) - len(x) - 1) * cumsum) / (len(x) * np.sum(x))
    return gini

def WAP(bid, ask, bid_size, ask_size):
    '''加权平均价格'''
    # 计算加权平均价格
    wap = ((bid[:, 0] * bid_size[:,0]) + (ask[:, 0] * ask_size[:,0])) / (bid_size[:,0] + ask_size[:,0])
    return [np.mean(wap), np.std(wap), _gini(wap)]

def RSpread(bid, ask):
    '''相对扩散率'''
    # 计算相对扩散率
    r_spread = (ask[:, 0] - bid[:, 0]) / (ask[:, 0] + bid[:, 0])
    return [np.mean(r_spread), np.std(r_spread), _gini(r_spread)]

def WAPbidDiff(bid, ask, bid_size, ask_size):
    '''加权平均价格与买一价的差值'''
    # 计算加权平均价格与买一价的差值
    wap = ((bid[:, 0] * bid_size[:,0]) + (ask[:, 0] * ask_size[:,0])) / (bid_size[:,0] + ask_size[:,0])
    wap_bid_diff = wap - bid[:, 0]
    return [np.mean(wap_bid_diff), np.std(wap_bid_diff), _gini(wap_bid_diff)]

def WAP_sq_rtn(bid, ask, bid_size, ask_size):
    '''加权平均价格的平方'''
    # 计算加权平均价格的平方
    wap = ((bid[:, 0] * bid_size[:,0]) + (ask[:, 0] * ask_size[:,0])) / (bid_size[:,0] + ask_size[:,0])
    wap_sq_rtn = wap ** 2
    return [ np.std(wap_sq_rtn), _gini(wap_sq_rtn)]

def Sspread(bid_size, ask_size):
    '''挂单量相对扩散率'''
    # 计算挂单量相对扩散率
    s_spread = (ask_size[:, 0] - bid_size[:, 0]) / (ask_size[:, 0] + bid_size[:, 0])
    return [np.mean(s_spread), np.std(s_spread), _gini(s_spread)]

def norm_ask_size(ask_size):
    '''标准化卖方挂单量'''
    ask_size = ask_size[:,0] / np.sum(ask_size[:,0])
    return [np.mean(ask_size), np.std(ask_size), _gini(ask_size)]

#  Price Jump Prediction in Limit Order Book

def EXPratio(bid_size, ask_size):
    '''指数价格差比率'''
    r = np.log(np.sum(np.exp(bid_size[-1,:])) / np.sum(np.exp(ask_size[-1,:])))
    return r

# bid = np.array([[3,2,1],[4,3,2],[5,4,3]])  # p/p0-1
# ask = np.array([[6,7,8],[7,8,9],[8,9,10]])
# bid_size = np.array([[1,2,3],[2,3,4],[3,4,5]])
# ask_size = np.array([[2,1,3],[3,2,4],[4,3,5]])
# volume = np.array([2,1,3])
# price = np.array([3,7,5])
# print(MPC_skew(bid, ask))