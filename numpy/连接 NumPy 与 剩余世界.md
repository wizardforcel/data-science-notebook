# 连接 NumPy 与 剩余世界

```py
# 来源：NumPy Cookbook 2e Ch4
```

## 使用缓冲区协议

```py
# 协议在 Python 中相当于接口
# 是一种约束
import numpy as np 
import Image 
# from PIL import Image (Python 3) 
import scipy.misc

lena = scipy.misc.lena() 
# Lena 是 512x512 的灰度图像
# 创建与 Lena 宽高相同的 RGBA 图像，全黑色
data = np.zeros((lena.shape[0], lena.shape[1], 4), dtype=np.int8) 
# 将 data 的不透明度设置为 Lena 的灰度
data[:,:,3] = lena.copy() 

# 将 data 转成 RGBA 的图像格式，并保存
img = Image.frombuffer("RGBA", lena.shape, data, 'raw', "RGBA", 0, 1) 
img.save('lena_frombuffer.png')

# 每个像素都设为 #FC0000FF （红色）
data[:,:,3] = 255 
data[:,:,0] = 222 
img.save('lena_modified.png') 
```

![](http://upload-images.jianshu.io/upload_images/118142-a97dc0a1e708c28e.jpg)

![](http://upload-images.jianshu.io/upload_images/118142-de1458a7284da12e.jpg)

## 数组协议

```py
from __future__ import print_function 
import numpy as np 
import Image 
import scipy.misc

# 获取上一节的第一个图像
lena = scipy.misc.lena() 
data = np.zeros((lena.shape[0], lena.shape[1], 4), dtype=np.int8) 
data[:,:,3] = lena.copy() 
img = Image.frombuffer("RGBA", lena.shape, data, 'raw', "RGBA", 0, 1) 

# 获取数组接口（协议），实际上它是个字典
array_interface = img.__array_interface__ 
print("Keys", array_interface.keys())
print("Shape", array_interface['shape']) 
print("Typestr", array_interface['typestr'])
'''
Keys ['shape', 'data', 'typestr'] 
Shape (512, 512, 4) 
Typestr |u1 
'''

# 将图像由 PIL.Image 类型转换回 np.array
numpy_array = np.asarray(img) 
print("Shape", numpy_array.shape) 
print("Data type", numpy_array.dtype)
'''
Shape (512, 512, 4) 
Data type uint8
''' 
```

## 与 Matlab 和 Octave 交换数据

```py
# 创建 0 ~ 6 的数组
a = np.arange(7) 
# 将 a 作为 array 保存在 a.mat 中
scipy.io.savemat("a.mat", {"array": a})
'''
octave-3.4.0:2> load a.mat 
octave-3.4.0:3> array 
array =
  0
  1
  ...
  6
'''

# 还可以再读取进来
mat = io.loadmat("a.mat")
print mat
# {'array': array([[0, 1, 2, 3, 4, 5, 6]]), '__version__': '1.0', '__header__': 'MATLAB 5.0 MAT-file Platform: nt, Created on: Sun Jun 11 18:48:29 2017', '__globals__': []}
```
