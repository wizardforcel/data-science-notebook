# 掌握 NumPy 常用函数

## 斐波那契数的第 n 项

```py
# 来源：NumPy Cookbook 2e Ch3.1

import numpy as np

# 斐波那契数列的每个新项都由之前的两项相加而成
# 以 1 和 2 开始，前 10 项为：
# 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...

# 斐波那契数列的通项公式为：
# fn = (phi ** n - (-phi) ** (-n)) / 5 ** 0.5
# 其中 phi 是黄金比例，phi = (1 + 5 ** 0.5) / 2

# 考虑一个斐波那契数列，每一项的值不超过四百万
# 求出值为偶数的项的和

# 1. 计算 phi
phi = (1 + np.sqrt(5))/2 
print("Phi", phi)
# Phi 1.61803398875

# 2. 寻找小于四百万的项的索引
n = np.log(4 * 10 ** 6 * np.sqrt(5) + 0.5)/np.log(phi) 
print(n)
# 33.2629480359

# 3. 创建 1 ~ n 的数组
n = np.arange(1, n) 
print(n)

# 4. 计算斐波那契数
fib = (phi**n - (-1/phi)**n)/np.sqrt(5) 
print("First 9 Fibonacci Numbers", fib[:9])
# First 9 Fibonacci Numbers [  1.   1.   2.   3.   5.   8.  13.  21.  34.] 

# 5. 转换为整数（可选）
fib = fib.astype(int) 
print("Integers", fib)
'''
Integers [      1       1       2       3       5       8      13      21      34
  ... snip ... snip ...
  317811  514229  832040 1346269 2178309 3524578] 
'''

# 6. 选择值为偶数的项
eventerms = fib[fib % 2 == 0] 
print(eventerms)
# [      2       8      34     144     610    2584   10946   46368  196418  832040 3524578]

# 7. 求和
print(eventerms.sum())
# 4613732
```

## 寻找质因数

```py
# 来源：NumPy Cookbook 2e Ch3.2

from __future__ import print_function 
import numpy as np

# 13195 的质因数是 5, 7, 13 和 29
# 600851475143 的最大质因数是多少呢？

N = 600851475143 
LIM = 10 ** 6


def factor(n): 
    # 1. 创建搜索范围的数组
    # a 是 sqrtn ~ sqrtn + lim - 1 的数组
    # 其中 sqrtn 是 n 平方根向上取整
    # lim 是 sqrtn 和 10e6 的较小值
    a = np.ceil(np.sqrt(n))   
    lim = min(n, LIM)   
    a = np.arange(a, a + lim)   
    b2 = a ** 2 - n
    
    # 2. 检查 b 是否是平方数
    # modf 用于取小数部分
    fractions = np.modf(np.sqrt(b2))[0]
    
    # 3. 寻找没有小数部分的地方
    # 这里的 b 为平方数
    indices = np.where(fractions == 0)
    
    # 4. 寻找 b 为平方数时，a 的值，并取出第一个
    a = np.ravel(np.take(a, indices))[0]
    # 或者 a = a[indices][0]
    
    # 求出 c 和 d
    a = int(a)   
    b = np.sqrt(a ** 2 - n)    
    b = int(b)   
    c = a + b   
    d = a - b
    
    # 到达终止条件则返回
    if c == 1 or d == 1:      
        return
    
    # 打印当前 c 和 d 并递归
    print(c, d)   
    factor(c)   
    factor(d)

factor(N)
'''
1234169 486847 
1471 839 
6857 71
'''
```

## 寻找回文数

```py
# 来源：NumPy Cookbook 2e Ch3.3

# 回文数正着读还是反着读都一样
# 由两个两位数的乘积构成的最大回文数是 9009 = 91 x 99
# 寻找两个三位数乘积构成的最大回文数

# 1. 创建三位数的数组
a = np.arange(100, 1000) 
np.testing.assert_equal(100, a[0]) np.testing.assert_equal(999, a[-1])

# 2. 创建两个数组中元素的乘积
# outer 计算数组的外积，也就是 a[i] x a[j] 的矩阵
# ravel 将其展开之后，就是每个元素乘积的数组了
numbers = np.outer(a, a) 
numbers = np.ravel(numbers) 
numbers.sort() 
np.testing.assert_equal(810000, len(numbers)) 
np.testing.assert_equal(10000, numbers[0]) 
np.testing.assert_equal(998001, numbers[-1])

#3. 寻找最大的回文数
for number in numbers[::-1]:   
    s = str(numbers[i])
    if s == s[::-1]:
        print(s)     
        break
```