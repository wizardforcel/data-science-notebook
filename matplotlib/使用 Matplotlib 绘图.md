# 使用 Matplotlib 绘图

```py
# 来源：NumPy Biginner's Guide 2e ch9
```

# 绘制多项式函数

```py
import numpy as np
import matplotlib.pyplot as plt

# 创建函数 func = x ** 3 + 2 * x ** 2 + 3 * x + 4
# poly1d 根据系数数组创建函数，高项系数在前
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
# x 值是 -10 ~ 10 取 30 个点
x = np.linspace(-10, 10, 30)
# 计算相应的 y 值
y = func(x)

# 绘制函数，plot 并不会立即显示
plt.plot(x, y)
# 设置两个轴的标签
plt.xlabel('x')
plt.ylabel('y(x)')
# 显示图像
plt.show()
```

![](http://upload-images.jianshu.io/upload_images/118142-d6ac4e0955e1c4cc.jpg)

## 绘制多项式函数及其导函数

```py
import numpy as np
import matplotlib.pyplot as plt

# func = x ** 3 + 2 * x ** 2 + 3 * x + 4
# func1 是它的导数，func' = 3 * x ** 2 + 4 * x + 3
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
func1 = func.deriv(m=1)
x = np.linspace(-10, 10, 30)
y = func(x)
y1 = func1(x)

# 将原函数绘制为红色的散点
# 导函数绘制为绿色的虚线
plt.plot(x, y, 'ro’, x, y1, 'g--’)
plt.xlabel('x’)
plt.ylabel('y’)
plt.show()
# 可以看到这里导函数的零点是原函数的驻点
```

![](http://upload-images.jianshu.io/upload_images/118142-5759936e318af72e.jpg)

## 分别绘制多项式函数及其导数

```py
import numpy as np
import matplotlib.pyplot as plt

# func = x ** 3 + 2 * x ** 2 + 3 * x + 4
func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
x = np.linspace(-10, 10, 30)
y = func(x)
# func1 是它的导数，func' = 3 * x ** 2 + 4 * x + 3
func1 = func.deriv(m=1)
y1 = func1(x)
# func2 是二阶导数，func'' = 6 * x + 4
func2 = func.deriv(m=2)
y2 = func2(x)

# 三行一列的第一个位置
plt.subplot(311)
# 将原函数绘制为红色曲线
plt.plot(x, y, 'r-’)
plt.title("Polynomial")

# 三行一列的第二个位置
plt.subplot(312)
# 将一阶导函数绘制为蓝色三角
plt.plot(x, y1, 'b^’)
plt.title("First Derivative")

# 三行一列的第三个位置
plt.subplot(313)
# 将一阶导函数绘制为绿色散点
plt.plot(x, y2, 'go’)
plt.title("Second Derivative")
plt.xlabel('x’)
plt.ylabel('y’)
plt.show()
```

![](http://upload-images.jianshu.io/upload_images/118142-baf9c5238e055da3.jpg)

## K 线图

```py
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
from matplotlib.dates import MonthLocator
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.finance import candlestick
import sys
from datetime import date
import matplotlib.pyplot as plt

today = date.today()
start = (today.year - 1, today.month, today.day)

alldays = DayLocator()              
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")

symbol = 'DISH’

if len(sys.argv) == 2:
   symbol = sys.argv[1]

# 导入 DISH 一年的股票数据
quotes = quotes_historical_yahoo(symbol, start, today)

# 获取 Figure 和 Axes 实例
fig = plt.figure()
ax = fig.add_subplot(111)
# 设置 Locator 和 Formatter
ax.xaxis.set_major_locator(months)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(month_formatter)

# candlestick 用于绘制 K 线图
# ax 是 Axes 实例
# quotes 是股票数据，行是记录，列是属性
candlestick(ax, quotes)
fig.autofmt_xdate()
plt.show()
```

![](http://upload-images.jianshu.io/upload_images/118142-528f3113f9b5f69c.jpg)

## 绘制收盘价分布图

```py
from matplotlib.finance import quotes_historical_yahoo
import sys
from datetime import date
import matplotlib.pyplot as plt
import numpy as np

today = date.today()
start = (today.year - 1, today.month, today.day)

symbol = 'DISH’

if len(sys.argv) == 2:
   symbol = sys.argv[1]

# 导入 DISH 一年的股票数据
# 行为记录，列为属性
# 分别为日期、开盘价、最高价、最低价、收盘价、成交量
quotes = quotes_historical_yahoo(symbol, start, today)
# 获取收盘价
quotes = np.array(quotes)
close = quotes.T[4]

# 绘制直方图，横轴是数据分布，纵轴是频数
# 第一个参数是数据，第二个参数是分组数量，默认为 10
plt.hist(close, np.sqrt(len(close)))
plt.show()
```

![](http://upload-images.jianshu.io/upload_images/118142-ea84aed4760be212.jpg)

## 绘制收益和成交量差值的散点图

```py
from matplotlib.finance import quotes_historical_yahoo
import sys
from datetime import date
import matplotlib.pyplot as plt
import numpy as np

today = date.today()
start = (today.year - 1, today.month, today.day)

symbol = 'DISH’

if len(sys.argv) == 2:
   symbol = sys.argv[1]

# 获取 DISH 一年的收盘价和成交量
quotes = quotes_historical_yahoo(symbol, start, today)
quotes = np.array(quotes)
close = quotes.T[4]
volume = quotes.T[5]
# 计算收益和成交量差值
ret = np.diff(close)/close[:-1]
volchange = np.diff(volume)/volume[:-1]

fig = plt.figure()
ax = fig.add_subplot(111)
# 绘制散点图，参数分别为横轴、纵轴、颜色、大小、透明度
# 横轴为收益，纵轴为成交量差值
# 颜色随收益变化，大小随成交量变化
ax.scatter(ret, volchange, c=ret * 100, s=volchange * 100, alpha=0.5)
ax.set_title('Close and volume returns’)
ax.grid(True)

plt.show()
```

![](http://upload-images.jianshu.io/upload_images/118142-1a0ba265896ac51f.jpg)
