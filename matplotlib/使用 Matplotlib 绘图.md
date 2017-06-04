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