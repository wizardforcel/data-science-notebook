# NumPy 中的线性代数

```py
# 来源：NumPy Essentials ch5
```

## 矩阵

```py
import numpy as np 
ndArray = np.arange(9).reshape(3,3) 
# matrix 可以从 ndarray 直接构建
x = np.matrix(ndArray) 
# identity 用于构建单位矩阵
y = np.mat(np.identity(3)) 
x 
'''
matrix([[0, 1, 2], 
        [3, 4, 5], 
        [6, 7, 8]]) 
'''
y 
''' 
matrix([[1., 0., 0.], 
        [0., 1., 0.], 
        [0., 0., 1.]]) 
'''

# 矩阵的相加是逐元素的
x + y
'''
matrix([[ 1.,  1.,  2.], 
        [ 3.,  5.,  5.], 
        [ 6.,  7.,  9.]]) 
'''

# 矩阵乘法是一行乘一列
x * x 
'''
matrix([[ 15,  18,  21], 
        [ 42,  54,  66], 
        [ 69,  90, 111]]) 
'''
# np.dot 效果相同
np.dot(ndArray, ndArray) 
'''
array([[ 15,  18,  21], 
       [ 42,  54,  66], 
       [ 69,  90, 111]]) 
'''

# 矩阵的乘方
x ** 3 
'''
matrix([[ 180,  234,  288], 
        [ 558,  720,  882], 
        [ 936, 1206, 1476]])
'''

z = np.matrix(np.random.random_integers(1, 50, 9).reshape(3,3)) 
z
'''
matrix([[32, 21, 28], 
        [ 2, 24, 22], 
        [32, 20, 22]]) 
'''

# I 属性求矩阵的逆，不可逆时报错
z.I 
'''
matrix( [[-0.0237 -0.0264  0.0566] 
         [-0.178   0.0518  0.1748] 
         [ 0.1963 -0.0086 -0.1958]]) 
'''

# H 属性求共轭转置
z.H
'''
matrix([[32  2 32] 
        [21 24 20] 
        [28 22 22]]) 
'''

# 求解线性方程组
# Ax = b => x = A.I b
# 可以从这种字符串创建矩阵
# 空格分隔行，分号分隔列
A = np.mat('3 1 4; 1 5 9; 2 6 5') 
b = np.mat([[1],[2],[3]]) 
x = A.I * b
x 
'''
matrix([[ 0.2667], 
        [ 0.4667], 
        [-0.0667]]) 
'''

# allclose 验证是否等价
np.allclose(A * x, b) 
# True 

# matrix 使用矩阵的方法来计算转置
x = np.arange(25000000).reshape(5000,5000) 
y = np.mat(x) 
'''
%timeit x.T 
10000000 loops, best of 3: 176 ns per loop 

%timeit y.T 
1000000 loops, best of 3: 1.36 µs per loop
'''

# A 属性返回等价的 ndarray
# A1 属性返回展开的等价 ndarray
A.A 
'''
array([[3, 1, 4], 
       [1, 5, 9], 
       [2, 6, 5]]) 
'''
A.A1 
# array([3, 1, 4, 1, 5, 9, 2, 6, 5]) 
```

## 线性代数

```py
x = np.array([[1, 2], [3, 4]]) 
y = np.array([[10, 20], [30, 40]]) 
# dot 对一维数组计算内积
# 对二维数组计算矩阵乘法
np.dot(x, y) 
'''
[[1*10+2*30, 1*20+2*40],
 [3*10+4*30, 3*20+4*40]]

=>

[[ 70, 100],
 [150, 220]] 
'''

# vdot 将数组展开计算内积
# 等价于 np.dot(x.ravel(), y.ravel())
np.vdot(x, y) 
# 1*10+2*20+3*30+4*40=300

# outer计算外积
# 返回一个二维数组，每个元素是 x[i]*y[j]
np.outer(x,y) 
'''
[[1*10, 1*20, 1*30, 1*40],
 [2*10, 2*20, 2*30, 2*40],
 [3*10, 3*20, 3*30, 3*40],
 [4*10, 4*20, 4*30, 4*40]]

=>
 
[[ 10,  20,  30,  40], 
 [ 20,  40,  60,  80], 
 [ 30,  60,  90, 120], 
 [ 40,  80, 120, 160]]
'''

# cross 计算叉积
# 接受一维二元或三元向量

a = np.array([1,0,0]) 
b = np.array([0,1,0]) 
np.cross(a,b) 
'''
det([[i, j, k],
     [1, 0, 0],
     [0, 1, 0]])
     
=> 

[0, 0, 1]
'''
np.cross(b,a) 
# array([ 0,  0, -1]) 

# linalg.det 计算行列式
# 对于二阶方阵来说
# det = m[0,0]*m[1,1] - m[0,1]*m[1,0]
x = np.array([[4,8],[7,9]]) 
np.linalg.det(x) 
# -20.000000000000007 

# linalg.inv 求逆
np.linalg.inv(x) 
'''
array([[-0.45,  0.4 ], 
       [ 0.35, -0.2 ]]) 
'''
np.mat(x).I 
'''
matrix([[-0.45,  0.4 ], 
        [ 0.35, -0.2 ]]) 
'''

# linalg.solve 解方程组
# 原理是 dot(inv(A), b)
# 不可逆时会报错
x = np.linalg.solve(A,b) 
x 
'''
matrix([[ 0.2667], 
        [ 0.4667], 
        [-0.0667]]) 
'''

## 分解

```py

x = np.random.randint(0, 10, 9).reshape(3,3) 
x
'''
array([[ 1,  5,  0] 
       [ 7,  4,  0] 
       [ 2,  9,  8]]) 
'''

# 特征值分解
# https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
# w 为特征值，v 的列向量为特征向量
w, v = np.linalg.eig(x) 
w
# array([ 8.,  8.6033,  -3.6033]) 
v
'''
array([[ 0.,  0.0384,  0.6834] 
       [ 0.,  0.0583, -0.6292] 
       [ 1.,  0.9976,  0.3702]] 
) 
'''

# 复矩阵的分解
```