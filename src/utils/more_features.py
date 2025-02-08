# 负责计算更多的因子

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





# bid = np.array([[3,2,1],[4,3,2],[5,4,3]])  
# ask = np.array([[6,7,8],[7,8,9],[8,9,10]])
# bid_size = np.array([[1,2,3],[2,3,4],[3,4,5]])
# ask_size = np.array([[2,1,3],[3,2,4],[4,3,5]])
# volume = np.array([2,1,3])
# price = np.array([3,7,5])




