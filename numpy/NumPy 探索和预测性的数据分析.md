# NumPy 探索和预测性的数据分析

```py
# 来源：NumPy Cookbook 2e Ch12
```

## 气压

```py
from __future__ import print_function 
import numpy as np import matplotlib.pyplot as plt 
from statsmodels.stats.adnorm import normal_ad

# 加载数据集
data = np.load('cbk12.npy')

# 取第一列，也就是平均气压
# 单位是 0.1hpa 转换为 hpa
meanp = .1 * data[:,1]

# 过滤掉 0 值
meanp = meanp[ meanp > 0]

# 获取描述性统计量
print("Max", meanp.max()) 
print("Min", meanp.min()) 
mean = meanp.mean() 
print("Mean", mean) 
print("Median", np.median(meanp)) 
std = meanp.std()
print("Std dev", std)
'''
Max 1048.3 
Min 962.1 
Mean 1015.14058231 
Median 1015.8 
Std dev 9.85889134337
'''

# 检测正态性
print("Normality", normal_ad(meanp)) 
# Normality (72.685781095773564, 0.0)

# 绘制平均气压直方图
plt.subplot(211) 
plt.title('Histogram of average atmospheric pressure') 
# bins 是横轴上的区间端点
# normed=True 表示纵轴是频率/区间长度，而不是频数
_, bins, _ = plt.hist(meanp, np.sqrt(len(meanp)), normed=True) 
# 绘制均值为 mean，标准差为 std 的正态函数
plt.plot(bins, 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean)**2/(2 * std**2)), 'r-', label="Gaussian PDF") 
plt.grid() 
plt.legend(loc='best') 
plt.xlabel('Average atmospheric pressure (hPa)') 
plt.ylabel('Frequency')

# 绘制平均气压的盒图
plt.subplot(212) 
plt.boxplot(meanp) 
plt.title('Boxplot of average atmospheric pressure') 
plt.ylabel('Average atmospheric pressure (hPa)') 
plt.grid()

# 密致布局
plt.tight_layout() 
plt.show()
```

![]()

## 每天的气压极差

```py
from __future__ import print_function 
import numpy as np 
import matplotlib.pyplot as plt 
import calendar as cal

data = np.load('cbk12.npy')

# 第二列是最高气压，第三列是最低气压
# 将单位由 0.1hpa 转换为 hpa
highs = .1 * data[:,2] 
lows = .1 * data[:,3]

# 过滤 0 值
highs[highs == 0] = np.nan 
lows[lows == 0] = np.nan

# 计算极差
ranges = highs - lows 

# 计算极差的各种统计量
print("Minimum daily range", np.nanmin(ranges)) 
print("Maximum daily range", np.nanmax(ranges))
print("Average daily range", np.nanmean(ranges)) 
print("Standard deviation", np.nanstd(ranges))
'''
Minimum daily range 0.4 
Maximum daily range 41.7 
Average daily range 6.11945360571 
Standard deviation 4.42162136692
'''

# 获得日期（第 0 列）
dates = data[:,0] 
# 格式是 YYYYMMDD
# 所以取后四位（% 1e4），然后再取前两位（/ 1e2）
months = (dates % 10000)/100 
# 过滤零值
months = months[~np.isnan(ranges)]
# 原书没有这句，导致后面ranges是未过滤的
# 下标无法对应
ranges = ranges[~np.isnan(ranges)]

# 按照月份来分组
monthly = [] 
month_range = np.arange(1, 13)

# 计算每个月份的气压极差
for month in month_range:
    indices = np.where(month == months)   
    monthly.append(np.nanmean(ranges[indices]))

# 绘制月份与平均气压极差的条形图
plt.bar(month_range, monthly) 
plt.title('Monthly average of daily pressure ranges') 
plt.xticks(month_range, cal.month_abbr[1:13]) 
plt.ylabel('Monthly Average (hPa)') 
plt.grid() 
plt.show()
```

![]()

## 年度气压均值

```py
import numpy as np 
import matplotlib.pyplot as plt

data = np.load('cbk12.npy')
# 获取均值、最高值和最低值
# 并转换为 hpa 单位
avgs = .1 * data[:,1] 
highs = .1 * data[:,2] 
lows = .1 * data[:,3]

# 过滤 0 值
avgs = np.ma.array(avgs, mask = avgs == 0) 
lows = np.ma.array(lows, mask = lows == 0) 
highs = np.ma.array(highs, mask = highs == 0)

# 获取年份
years = data[:,0]/10000

# 初始化按年份分组的数组
y_range = np.arange(1901, 2014) 
nyears = len(y_range) 
y_avgs = np.zeros(nyears)
y_highs = np.zeros(nyears) 
y_lows = np.zeros(nyears)

# 计算每一年的均值、最高值和最低值
for year in y_range:
    indices = np.where(year == years)
    y_avgs[year - 1901] = np.mean(avgs[indices])   
    y_highs[year - 1901] = np.max(highs[indices])   
    y_lows[year - 1901] = np.min(lows[indices])

plt.title('Annual atmospheric pressure for De Bilt(NL)') 
plt.ticklabel_format(useOffset=900, axis='y')

# 绘制均值、最高值和最低值
plt.plot(y_range, y_avgs, label='Averages')
h_mask = np.isfinite(y_highs) 
plt.plot(y_range[h_mask], y_highs[h_mask], '^', label='Highs')
l_mask = np.isfinite(y_lows) 
plt.plot(y_range[l_mask], y_lows[l_mask], 'v', label='Lows')

plt.xlabel('Year') 
plt.ylabel('Atmospheric pressure (hPa)') 
plt.grid() 
plt.legend(loc='best') 
plt.show()
```

![]()