# NumPy 音频和图像处理

```py
# 来源：NumPy Cookbook 2e Ch5
```

## 将图像加载进内存

```py
import numpy as np 
import matplotlib.pyplot as plt

# 首先生成一个 512x512 的图像
# 在里面画 30 个正方形
N = 512 
NSQUARES = 30

# 初始化
img = np.zeros((N, N), np.uint8) 
# 正方形的中心是 0 ~ N 的随机数
centers = np.random.random_integers(0, N, size=(NSQUARES, 2))
# 正方形的边长是 0 ~ N/9 的随机数
radii = np.random.randint(0, N/9, size=NSQUARES) 
# 颜色是 100 ~ 255 的随机数
colors = np.random.randint(100, 255, size=NSQUARES)

# 生成正方形
for i in xrange(NSQUARES):
    # 为每个正方形生成 x 和 y 坐标
    xindices = range(centers[i][0] - radii[i], centers[i][0]  + radii[i])   
    xindices = np.clip(xindices, 0, N - 1)   
    yindices = range(centers[i][1] - radii[i], centers[i][1]  + radii[i])   
    
    # clip 过滤范围之外的值
    # 相当于 yindices = yindices[(0 < yindices) & (yindices < N - 1)]
    yindices = np.clip(yindices, 0, N - 1)
    if len(xindices) == 0 or len(yindices) == 0:
        continue
    # 将 x 和 y 坐标转换成网格
    # 如果不转换成网格，只会给对角线着色
    coordinates = np.meshgrid(xindices, yindices)     
    img[coordinates] = colors[i]
   
# tofile 以二进制保存数组的内容，没有形状和类型信息 
img.tofile('random_squares.raw') 
# np.memmap 以二进制加载数组，如果类型不是 uint8，则需要执行
# 如果数组不是一维，还需要指定形状
img_memmap = np.memmap('random_squares.raw', shape=img.shape)

# 显示图像（会自动将灰度图映射为伪彩色）
plt.imshow(img_memmap) 
plt.axis('off') 
plt.show()
```

![]()

## 组合图像

```py
import numpy as np import 
matplotlib.pyplot as plt 
from scipy.misc import lena

ITERATIONS = 10 
lena = lena() 
SIZE = lena.shape[0] 
MAX_COLOR = 255. 
x_min, x_max = -2.5, 1 
y_min, y_max = -1, 1

# 数组初始化
x, y = np.meshgrid(np.linspace(x_min, x_max, SIZE),
                   np.linspace(y_min, y_max, SIZE)) 
c = x + 1j * y 
z = c.copy() 
fractal = np.zeros(z.shape, dtype=np.uint8) + MAX_COLOR 

# 生成 mandelbrot 图像 
for n in range(ITERATIONS):
    mask = np.abs(z) <= 4
    z[mask] = z[mask] ** 2 +  c[mask]
    fractal[(fractal == MAX_COLOR) & (-mask)] = (MAX_COLOR - 1) * n / ITERATIONS

# 绘制 mandelbrot 图像 
plt.subplot(211) 
plt.imshow(fractal) 
plt.title('Mandelbrot') 
plt.axis('off')

# 将 mandelbrot 和 lena 组合起来
plt.subplot(212) 
# choose 的作用是，如果 fractal 的元素小于 lena 的对应元素
# 就选择 fractal，否则选择 lena
# 相当于 np.fmin(fractal, lena)
plt.imshow(np.choose(fractal < lena, [fractal, lena])) 
plt.axis('off') 
plt.title('Mandelbrot + Lena')
plt.show()
```

![]()

## 使图像变模糊

```py
import numpy as np 
import matplotlib.pyplot as plt 
from random import choice 
import scipy 
import scipy.ndimage

# Initialization 
NFIGURES = 5 
k = np.random.random_integers(1, 5, NFIGURES) 
a = np.random.random_integers(1, 5, NFIGURES)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# 绘制原始 的 lena 图像
lena = scipy.misc.lena() 
plt.subplot(211) 
plt.imshow(lena) 
plt.axis('off')

# 绘制模糊的 lena 图像
plt.subplot(212) 
# 使用 sigma=4 的高斯过滤器
blurred = scipy.ndimage.gaussian_filter(lena, sigma=4)
plt.imshow(blurred) 
plt.axis('off')

# 在极坐标中绘图
# 极坐标无视 subplot
theta = np.linspace(0, k[0] * np.pi, 200) 
plt.polar(theta, np.sqrt(theta), choice(colors))

for i in xrange(1, NFIGURES):
    theta = np.linspace(0, k[i] * np.pi, 200)   
    plt.polar(theta, a[i] * np.cos(k[i] * theta), choice(colors))
plt.axis('off')
plt.show()
```

![]()