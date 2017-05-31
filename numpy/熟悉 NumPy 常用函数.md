# 熟悉 NumPy 常用函数

```py
# 来源：NumPy Biginner's Guide 2e ch3
```

## 读写文件

```py
import numpy as np

# eye 用于创建单位矩阵
i2 = np.eye(2)
print i2
'''
[[ 1.  0.]
[ 0.  1.]]
'''

# 将数组以纯文本保存到 eye.txt 中
np.savetxt("eye.txt", i2)
'''
eye.txt:
1.000000000000000000e+00 0.000000000000000000e+00
0.000000000000000000e+00 1.000000000000000000e+00
'''

# 还可以读进来
print np.loadtxt('eye.txt')
[[ 1.  0.]
 [ 0.  1.]]
```

## 读取 CSV

```py
'''
data.csv:
AAPL,28-01-2011, ,344.17,344.4,333.53,336.1,21144800
分别为：
名称，日期，空，开盘，最高，最低，收盘，成交量
'''

# delimiter 是分隔符，设置为 ','
# usecols 设置需要取的列，这里只选择了收盘和成交量
# unpack 设置为 True，返回的数组是以列为主
# 可以分别将收盘和成交量赋给 c 和 v
c, v = np.loadtxt('data.csv', delimiter=',', usecols=(6,7), unpack=True)
```

## 均值

```py
import numpy as np

c, v = np.loadtxt('data.csv', delimiter=',', usecols=(6,7), unpack=True)

# 计算成交量加权均价
# average 用于计算均值
# weights 参数指定权重
vwap = np.average(c, weights=v)
print "VWAP =", vwap
# VWAP = 350.589549353

# mean 函数也能用于计算均值
print "mean =", np.mean(c)
# mean =  351.037666667

# 计算时间时间加权均价
t = np.arange(len(c))
print "twap =", np.average(c, weights=t)
# twap = 352.428321839
```

## 最大最小值

```
import numpy as np

# 这次读入了最高价和最低价
h, l = np.loadtxt('data.csv', delimiter=',', usecols=(4,5), unpack=True)

# 计算历史最高价和最低价
print "highest =", np.max(h)
# highest = 364.9
print "lowest =", np.min(l)
# lowest = 333.53

# ptp 函数用于计算极差
print "Spread high price", np.ptp(h)
# Spread high price 24.86
print "Spread low price", np.ptp(l)
# Spread low price 26.97
```

## 简单统计

```py
import numpy as np

# 读入收盘价
c = np.loadtxt('data.csv', delimiter=',', usecols=(6,), unpack=True)

# 计算中位数
print "median =", np.median(c)
# median = 352.055

# 手动计算中位数
# 首先排个序
sorted_close = np.msort(c)
print "sorted =", sorted_close
# 然后取中间元素
N = len(c)
print "middle =", sorted[(N - 1)/2]
# middle = 351.99
# 由于我们的数组长度是偶数
# 中位数应该是中间两个数的均值
# print "average middle =", (sorted[N /2] + sorted[(N - 1) / 2]) / 2
# average middle = 352.055

# 方差
print "variance =", np.var(c)
# variance = 50.1265178889

# 手动计算方差
print "variance from definition =", np.mean((c - c.mean())**2)
# variance from definition = 50.1265178889
```

## 股票收益

```py
import numpy as np

# 简单收益
# 当天收盘价减去前一天收盘价，再除以前一天收盘价
# returns = np.diff( arr ) / arr[ : -1]

# 我们计算一下标准差（方差的平方根）
print "Standard deviation =", np.std(returns)
# Standard deviation = 0.0129221344368

# 对数收益
# 当天收盘价的对数前前一天收盘价的对数
logreturns = np.diff( np.log(c) )

# 计算收益为正的下标
# where 将布尔索引变成位置索引
posretindices = np.where(returns > 0)
print "Indices with positive returns", posretindices
# Indices with positive returns (array([ 0,  1,  4,  5,  6,  7,  9, 10, 11, 12, 16, 17, 18, 19, 21, 22, 23, 25, 28]),)

# 年化波动
annual_volatility = np.std(logreturns)/np.mean(logreturns)
annual_volatility = annual_volatility / np.sqrt(1./252.) 
print annual_volatility

# 月化波动
print "Monthly volatility", annual_volatility * np.sqrt(1./12.)
```

## 处理日期

```py
import numpy as np
from datetime import datetime

# 将日期映射为星期
# Monday 0
# Tuesday 1
# Wednesday 2
# Thursday 3
# Friday 4
# Saturday 5
# Sunday 6
def datestr2num(s):
    return datetime.strptime(s, "%d-%m-%Y").date().weekday()

# 读取星期和收盘价，converters 将日期映射成星期
dates, close = np.loadtxt('data.csv', delimiter=',', usecols=(1,6), converters={1: datestr2num}, unpack=True)
print "Dates =", dates
# Dates = [ 4.  0.  1.  2.  3.  4.  0.  1.  2.  3.  4.  0.  1.  2.  3.  4.  1.  2.  4.  0.  1.  2.  3.  4.  0.  1.  2.  3.  4.]

# 计算一周中每一天的均值
averages = np.zeros(5)

for i in range(5):
    indices = np.where(dates == i) 
    prices = np.take(close, indices)
    avg = np.mean(prices)
    print "Day", i, "prices", prices, "Average", avg
    averages[i] = avg
'''
Day 0 prices [[ 339.32  351.88  359.18  353.21  355.36]] Average 351.79
Day 1 prices [[ 345.03  355.2   359.9   338.61  349.31  355.76]] Average 350.635
Day 2 prices [[ 344.32  358.16  363.13  342.62  352.12  352.47]] Average 352.136666667
Day 3 prices [[ 343.44  354.54  358.3   342.88  359.56  346.67]] Average 350.898333333
Day 4 prices [[ 336.1   346.5   356.85  350.56  348.16  360.    351.99]] Average 350.022857143
'''

# 计算星期几最高，星期几最低
top = np.max(averages)
print "Highest average", top
# Highest average 352.136666667
print "Top day of the week", np.argmax(averages)
# Top day of the week 2

bottom = np.min(averages)
print "Lowest average", bottom
# Lowest average 350.022857143
print "Bottom day of the week", np.argmin(averages
# Bottom day of the week 4
```
