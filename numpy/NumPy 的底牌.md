# NumPy 的底牌

```py
# 来源：NumPy Cookbook 2e Ch11

np.random.seed(44) 
a = np.random.random_integers(-4, 4, 7) 
print(a) 
# [ 0 -1 -3 -1 -4  0 -1]

# ufunc 的 at 方法可以对数组元素部分调用
np.sign.at(a, [2, 4]) 
print(a) 
# np.sign.at(a, [2, 4]) print(a) 

np.random.seed(20) 
a = np.random.random_integers(0, 7, 9) 
print(a) 
# [3 2 7 7 4 2 1 4 3] 

# partition 仅仅排序所选位置
# 也就是说 a 中下标为 4 的元素在排序后的位置
# 其它的不保证
print(np.partition(a, 4)) 
# [2 3 1 2 3 7 7 4 4]

np.random.seed(46) 
a = np.random.randn(30) 
estimates = np.zeros((len(a), 3))

# nanmean nanvar 和 nanstd 可以用于计算
# 排除 NaN 值的均值、方差和标准差
for i in xrange(len(a)):
    # 依次把 a[i] 设为 NaN
    # 计算均值、方差和标准差
    b = a.copy()
    b[i] = np.nan
    estimates[i,] = [np.nanmean(b), np.nanvar(b), np.nanstd(b)]

print("Estimator variance", estimates.var(axis=0))
# Estimator variance [ 0.00079905  0.00090129  0.00034604]

# full 创建纯量数组
# full(size, val) 等价于 ones(size) * val
print(np.full((1, 2), 7)) 
# array([[ 7.,  7.]])

print(np.full((1, 2), 7, dtype=np.int)) 
array([[7, 7]])

a = np.linspace(0, 1, 5) 
print(a) 
# array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]) 

# full_like 接受数组，并取形状
# full_like(arr, val) 等价于 ones(arr.shape) * val
print(np.full_like(a, 7)) 
# array([ 7.,  7.,  7.,  7.,  7.])

print(np.full_like(a, 7, dtype=np.int)) 
# array([7, 7, 7, 7, 7])
```

## np.random.choice 随机选取

```py

# 取 400 个随机数，满足 B(5, 0.5)
N = 400 
np.random.seed(28) 
data = np.random.binomial(5, .5, size=N) 

# 从随机数中随机取 400x30 个值
# 等价于选取 Nx30 次
# 每次使用 randint(0, len(data)) 来生成下标
bootstrapped = np.random.choice(data, size=(N, 30)) 

# 计算每列的均值
means = bootstrapped.mean(axis=0) 

# 绘制盒图（包含最大值、最小值、中位数、两个四分位数）
plt.title('Bootstrapping demo') 
plt.grid() 
plt.boxplot(means) 
plt.plot(3 * [data.mean()], lw=3, label='Original mean') 
plt.legend(loc='best') 
plt.show()
```

![](http://upload-images.jianshu.io/upload_images/118142-903a8f2a3fb2f7e5.jpg)

## datetime64 类型

```
import numpy as np

# 由年月日构造
print(np.datetime64('2015-05-21')) 
# numpy.datetime64('2015-05-21')

# 去掉横杠
print(np.datetime64('20150521')) 
# 由年月构造
print(np.datetime64('2015-05')) 
# numpy.datetime64('20150521') 
# numpy.datetime64('2015-05')

# 由日期和时间构造
local = np.datetime64('1578-01-01T21:18') 
print(local) 
# numpy.datetime64('1578-01-01T21:18Z')

# 可以带上偏移
with_offset = np.datetime64('1578-01-01T21:18-0800') 
print(with_offset) 
# numpy.datetime64('1578-01-02T05:18Z')

# datetime64 作差会生成 timedelta64
print(local - with_offset)
# numpy.timedelta64(-480,'m') 
```