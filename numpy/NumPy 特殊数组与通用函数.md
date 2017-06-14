# NumPy 特殊数组与通用函数

```py
# 来源：NumPy Cookbook 2e ch6
```

## 创建通用函数

```py
from __future__ import print_function 
import numpy as np

# 我们需要定义对单个元素操作的函数
def double(a):
    return 2 * a

# frompyfunc（或者 vectorize）
# 将其转换为对数组每个元素操作的函数
ufunc = np.frompyfunc(double, 1, 1) 
print("Result", ufunc(np.arange(4)))
# Result [0 2 4 6]
```

## 勾股数

```py
from __future__ import print_function 
import numpy as np

# 勾股数是指满足 a ** 2 + b ** 2 == c ** 2 的三个数
# 我们使 a = m ** 2 - n ** 2，b = 2 * m * n
# c = m ** 2 + n ** 2，来寻找 a + b + c == 1000 的勾股数

# m 和 n 都取 0 ~ 32
m = np.arange(33) 
n = np.arange(33) 

# 计算 a，b 和 c
# outer 生成 a[i] op b[j] 为每个元素的矩阵
# 相当于 meshgrid 之后再逐元素操作
a = np.subtract.outer(m ** 2, n ** 2) 
b = 2 * np.multiply.outer(m, n) 
c = np.add.outer(m ** 2, n ** 2)

# 取符合我们条件的下标
# where 把布尔下标转换为位置下标
idx =  np.where((a + b + c) == 1000) 

# 验证并打印结果
np.testing.assert_equal(a[idx]**2 + b[idx]**2, c[idx]**2) 
print(a[idx], b[idx], c[idx]) 
# [375] [200] [425]
```

## CharArray 字符串操作

```py
# chararray 数组的元素只能是字符串
# 并且拥有许多字符串专用的方法
# 虽然我们可以为字符串创建通用函数
# 但是直接使用这些方法更省事

import urllib2 
import numpy as np 
import re

# 使用 urllib2 库下载网页
# 更推荐 requests 库
response = urllib2.urlopen('http://python.org/') 
html = response.read() 

# 替换掉所有标签
html = re.sub(r'<.*?>', '', html) 

# 创建仅仅包含该 HTML 的一维数组
# 并转为 chararray
carray = np.array(html).view(np.chararray) 

# expandtabs 将 TAB 转换为指定个数的空格
carray = carray.expandtabs(1) 
# splitlines 按换行符分割，会多一个维度
carray = carray.splitlines() 
print(carray)
```

## 创建屏蔽数组

```py
from __future__ import print_function 
import numpy as np from scipy.misc 
import lena 
import matplotlib.pyplot as plt

# 加载 Lena 图像
lena = lena() 

# 掩码数组和图像形状一致，元素取 0 和 1 的随机数
random_mask = np.random.randint(0, 2, size=lena.shape)

# 绘制原始图像
plt.subplot(221) 
plt.title("Original") 
plt.imshow(lena) 
plt.axis('off')

# ma.array 创建屏蔽数组
# 如果 random_mask 中某个元素是 0
# masked_array 中就将其屏蔽
# 访问会返回 masked
# 但是转换回 np.array 时会恢复
masked_array = np.ma.array(lena, mask=random_mask)
print(masked_array) 

# 绘制掩码后的图像
plt.subplot(222) 
plt.title("Masked") 
plt.imshow(masked_array) 
plt.axis('off')
```

![]()

## 忽略负数以及极值

```py
from __future__ import print_function 
import numpy as np 
from matplotlib.finance 
import quotes_historical_yahoo 
from datetime import date 
import matplotlib.pyplot as plt

def get_close(ticker):
    # 获取指定股票近一年的收盘价
    today = date.today()
    start = (today.year - 1, today.month, today.day)
    quotes = quotes_historical_yahoo(ticker, start, today)
    return np.array([q[4] for q in quotes])

# 获取 AAPL 一年的收盘价
close = get_close('AAPL')

triples = np.arange(0, len(close), 3) 
print("Triples", triples[:10], "...")
# Triples [ 0  3  6  9 12 15 18 21 24 27] ... 

# 创建等长的全 1 数组
signs = np.ones(len(close)) 
print("Signs", signs[:10], "...")
# Signs [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.] ... 

# sign 中每隔三个元素变为 -1
signs[triples] = -1 
print("Signs", signs[:10], "...")
# Signs [-1.  1.  1. -1.  1.  1. -1.  1.  1. -1.] ...

# ma.log 的作用是
# 如果元素小于等于 0，将其屏蔽
# 如果元素大于 0，取对数
ma_log = np.ma.log(close * signs) 
print("Masked logs", ma_log[:10], "...")
# Masked logs [-- 5.93655586575 5.95094223368 -- 5.97468290742 5.97510711452 -- 6.01674381162 5.97889061623 --] ...

dev = close.std() 
avg = close.mean() 
# 屏蔽 avg - dev 到 avg + dev 之外的元素
inside = np.ma.masked_outside(close, avg - dev, avg + dev) 
print("Inside", inside[:10], "...")
# Inside [-- -- -- -- -- -- 409.429675172    410.240597855 -- --] ...

# 绘制原始数据
plt.subplot(311) 
plt.title("Original") 
plt.plot(close)

# 绘制对数屏蔽后的数据
plt.subplot(312) 
plt.title("Log Masked") 
plt.plot(np.exp(ma_log))

# 绘制范围屏蔽后的数据
plt.subplot(313) 
plt.title("Not Extreme") 
plt.plot(inside)

plt.tight_layout() 
plt.show()
```

![]()

## 记录数组

```py
# rec.array 是 array 的子类
# 可以通过元素的属性来访问元素
from __future__ import print_function 
import numpy as np from matplotlib.finance 
import quotes_historical_yahoo 
from datetime import date

tickers = ['MRK', 'T', 'VZ']

def get_close(ticker):
    # 获取指定股票近一年的收盘价
    today = date.today()
    start = (today.year - 1, today.month, today.day)
    quotes = quotes_historical_yahoo(ticker, start, today)
    return np.array([q[4] for q in quotes])
    
# 创建记录数组，来统计每个股票的代码、
# 标准分（标准差的倒数）、均值和得分
weights = np.recarray((len(tickers),), dtype=[('symbol', np.str_, 16),
    ('stdscore', float), ('mean', float), ('score', float)])

for i, ticker in enumerate(tickers):
    # 获取收盘价、计算对数收益
    close = get_close(ticker)
    logrets = np.diff(np.log(close))
    # 保存符号、对数收益的均值和标准分
    weights[i]['symbol'] = ticker
    weights[i]['mean'] = logrets.mean()   
    weights[i]['stdscore'] = 1/logrets.std()
    weights[i]['score'] = 0

# 每个股票的均值和标准分需要除以相应的总数
for key in ['mean', 'stdscore']:
    wsum = weights[key].sum()
    weights[key] = weights[key]/wsum

# 得分是标准分和均值的均值
weights['score'] = (weights['stdscore'] + weights['mean'])/2 weights['score'].sort()

# 打印每个股票的信息
for record in weights:
    print("%s,mean=%.4f,stdscore=%.4f,score=%.4f" % (record['symbol'], record['mean'], record['stdscore'], record['score']))
'''
MRK,mean=0.8185,stdscore=0.2938,score=0.2177 
T,mean=0.0927,stdscore=0.3427,score=0.2262 
VZ,mean=0.0888,stdscore=0.3636,score=0.5561 
'''
```