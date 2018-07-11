# TensorFlow 高效编程

> 原文：[vahidk/EffectiveTensorflow](https://github.com/vahidk/EffectiveTensorflow)

> 译者：[一译翻译组](https://yiyibooks.cn/wizard/effective-tf/index.html)、[飞龙](https://github.com/wizardforcel)

> 
协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)

## 一、TensorFlow 基础

TensorFlow 和其他数字计算库（如 numpy）之间最明显的区别在于 TensorFlow 中操作的是符号。这是一个强大的功能，这保证了 TensorFlow 可以做很多其他库（例如 numpy）不能完成的事情（例如自动区分）。这可能也是它更复杂的原因。今天我们来一步步探秘 TensorFlow，并为更有效地使用 TensorFlow 提供了一些指导方针和最佳实践。

我们从一个简单的例子开始，我们要乘以两个随机矩阵。首先我们来看一下在 numpy 中如何实现：

```py
import numpy as np
x = np.random.normal(size=[10, 10])
y = np.random.normal(size=[10, 10])
z = np.dot(x, y)
print(z)

```

现在我们使用 TensorFlow 中执行完全相同的计算：  

```py
import TensorFlow as tf
x = tf.random_normal([10, 10])
y = tf.random_normal([10, 10])
z = tf.matmul(x, y)
sess = tf.Session()
z_val = sess.run(z)
print(z_val)

```

与立即执行计算并将结果复制给输出变量`z`的 numpy 不同，TensorFlow 只给我们一个可以操作的张量类型。如果我们尝试直接打印`z`的值，我们得到这样的东西：  

```py
Tensor("MatMul:0", shape=(10, 10), dtype=float32)
```

由于两个输入都是已经定义的类型，TensorFlow 能够推断张量的符号及其类型。为了计算张量的值，我们需要创建一个会话并使用`Session.run`方法进行评估。

要了解如此强大的符号计算到底是什么，我们可以看看另一个例子。假设我们有一个曲线的样本（例如`f(x)= 5x ^ 2 + 3`），并且我们要估计`f(x)`在不知道它的参数的前提下。我们定义参数函数为`g(x，w)= w0 x ^ 2 + w1 x + w2`，它是输入`x`和潜在参数`w`的函数，我们的目标是找到潜在参数，使得`g(x, w)≈f(x)`。这可以通过最小化损失函数来完成：`L(w)=(f(x)-g(x，w))^ 2`。虽然这问题有一个简单的封闭式的解决方案，但是我们选择使用一种更为通用的方法，可以应用于任何可以区分的任务，那就是使用随机梯度下降。我们在一组采样点上简单地计算相对于`w`的`L(w)`的平均梯度，并沿相反方向移动。

以下是在 TensorFlow 中如何完成：  

```py
import numpy as np
import TensorFlow as tf
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.get_variable("w", shape=[3, 1])
f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
yhat = tf.squeeze(tf.matmul(f, w), 1)
loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
    print(loss_val)
print(sess.run([w]))

```

通过运行这段代码，我们可以看到下面这组数据：

```
[4.9924135, 0.00040895029, 3.4504161]
```

这与我们的参数已经相当接近。

这只是 TensorFlow 可以做的冰山一角。许多问题，如优化具有数百万个参数的大型神经网络，都可以在 TensorFlow 中使用短短的几行代码高效地实现。而且 TensorFlow 可以跨多个设备和线程进行扩展，并支持各种平台。

## 二、理解静态和动态形状

在 **TensorFlow** 中，`tensor`有一个在图构建过程中就被决定的**静态形状属性**， 这个静态形状可以是**未规定的**，比如，我们可以定一个具有形状`[None, 128]`大小的`tensor`。

```python
import TensorFlow as tf
a = tf.placeholder(tf.float32, [None, 128])
```

这意味着`tensor`的第一个维度可以是任何尺寸，这个将会在`Session.run()`中被动态定义。当然，你可以查询一个`tensor`的静态形状，如：

```python
static_shape = a.shape.as_list()  # returns [None, 128]
```

为了得到一个`tensor`的动态形状，你可以调用`tf.shape`操作，这将会返回指定tensor的形状，如：

```python
dynamic_shape = tf.shape(a)
```

`tensor`的静态形状可以通过方法`Tensor_name.set_shape()`设定，如：

```python
a.set_shape([32, 128])  # static shape of a is [32, 128]
a.set_shape([None, 128])  # first dimension of a is determined dynamically
```

调用`tf.reshape()`方法，你可以动态地重塑一个`tensor`的形状，如：

```python
a =  tf.reshape(a, [32, 128])
```

可以定义一个函数，当静态形状的时候返回其静态形状，当静态形状不存在时，返回其动态形状，如：

```python
def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims
```

现在，如果我们需要将一个三阶的`tensor`转变为 2 阶的`tensor`，通过折叠第二维和第三维成一个维度，我们可以通过我们刚才定义的`get_shape()`方法进行，如：

```python
b = tf.placeholder(tf.float32, [None, 10, 32])
shape = get_shape(b)
b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
```

注意到无论这个`tensor`的形状是静态指定的还是动态指定的，这个代码都是有效的。事实上，我们可以写出一个通用的`reshape`函数，用于折叠维度的任意列表:

```python
import TensorFlow as tf
import numpy as np

def reshape(tensor, dims_list):
  shape = get_shape(tensor)
  dims_prod = []
  for dims in dims_list:
    if isinstance(dims, int):
      dims_prod.append(shape[dims])
    elif all([isinstance(shape[d], int) for d in dims]):
      dims_prod.append(np.prod([shape[d] for d in dims]))
    else:
      dims_prod.append(tf.prod([shape[d] for d in dims]))
  tensor = tf.reshape(tensor, dims_prod)
  return tensor
```

然后折叠第二个维度就变得特别简单了。

```python
b = tf.placeholder(tf.float32, [None, 10, 32])
b = reshape(b, [0, [1, 2]])
```

## 三、作用域和何时使用它

在 TensorFlow 中，变量和张量有一个名字属性，用于作为他们在图中的标识。如果你在创造变量或者张量的时候，不给他们显式地指定一个名字，那么 TF 将会自动地，隐式地给他们分配名字，如：

```python
a = tf.constant(1)
print(a.name)  # prints "Const:0"

b = tf.Variable(1)
print(b.name)  # prints "Variable:0"
```

你也可以在定义的时候，通过显式地给变量或者张量命名，这样将会重写他们的默认名，如：

```python
a = tf.constant(1, name="a")
print(a.name)  # prints "b:0"

b = tf.Variable(1, name="b")
print(b.name)  # prints "b:0"
```

TF 引进了两个不同的上下文管理器，用于更改张量或者变量的名字，第一个就是`tf.name_scope`，如：

```python
with tf.name_scope("scope"):
  a = tf.constant(1, name="a")
  print(a.name)  # prints "scope/a:0"

  b = tf.Variable(1, name="b")
  print(b.name)  # prints "scope/b:0"

  c = tf.get_variable(name="c", shape=[])
  print(c.name)  # prints "c:0"
```

我们注意到，在 TF 中，我们有两种方式去定义一个新的变量，通过`tf.Variable()`或者调用`tf.get_variable()`。在调用`tf.get_variable()`的时候，给予一个新的名字，将会创建一个新的变量，但是如果这个名字并不是一个新的名字，而是已经存在过这个变量作用域中的，那么就会抛出一个`ValueError`异常，意味着重复声明一个变量是不被允许的。

`tf.name_scope()`只会影响到**通过调用`tf.Variable`创建的**张量和变量的名字，而**不会影响到通过调用`tf.get_variable()`创建**的变量和张量。  

和`tf.name_scope()`不同，`tf.variable_scope()`也会修改，影响通过`tf.get_variable()`创建的变量和张量，如：

```python
with tf.variable_scope("scope"):
  a = tf.constant(1, name="a")
  print(a.name)  # prints "scope/a:0"

  b = tf.Variable(1, name="b")
  print(b.name)  # prints "scope/b:0"

  c = tf.get_variable(name="c", shape=[])
  print(c.name)  # prints "scope/c:0"
with tf.variable_scope("scope"):
  a1 = tf.get_variable(name="a", shape=[])
  a2 = tf.get_variable(name="a", shape=[])  # Disallowed
```

但是如果我们真的想要重复使用一个先前声明过了变量怎么办呢？变量管理器同样提供了一套机制去实现这个需求：

```python
with tf.variable_scope("scope"):
  a1 = tf.get_variable(name="a", shape=[])
with tf.variable_scope("scope", reuse=True):
  a2 = tf.get_variable(name="a", shape=[])  # OK
This becomes handy for example when using built-in neural network layers:

features1 = tf.layers.conv2d(image1, filters=32, kernel_size=3)
# Use the same convolution weights to process the second image:
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
  features2 = tf.layers.conv2d(image2, filters=32, kernel_size=3)
```

这个语法可能看起来并不是特别的清晰明了。特别是，如果你在模型中想要实现一大堆的变量共享，你需要追踪各个变量，比如说什么时候定义新的变量，什么时候要复用他们，这些将会变得特别麻烦而且容易出错，因此 TF 提供了 TF 模版自动解决变量共享的问题：

```python
conv3x32 = tf.make_template("conv3x32", lambda x: tf.layers.conv2d(x, 32, 3))
features1 = conv3x32(image1)
features2 = conv3x32(image2)  # Will reuse the convolution weights.
```

你可以将任何函数都转换为 TF 模版。当第一次调用这个模版的时候，在这个函数内声明的变量将会被定义，同时在接下来的连续调用中，这些变量都将自动地复用。

## 四、广播的优缺点

TensorFlow 支持广播机制，可以广播逐元素操作。正常情况下，当你想要进行一些操作如加法，乘法时，你需要确保操作数的形状是相匹配的，如：你不能将一个具有形状`[3, 2]`的张量和一个具有`[3,4]`形状的张量相加。但是，这里有一个特殊情况，那就是当你的其中一个操作数是一个某个维度为一的张量的时候，TF 会隐式地填充它的单一维度方向，以确保和另一个操作数的形状相匹配。所以，对一个`[3,2]`的张量和一个`[3,1]`的张量相加在 TF 中是合法的。

```python
import TensorFlow as tf

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[1.], [2.]])
# c = a + tf.tile(b, [1, 2])
c = a + b
```

广播机制允许我们在隐式情况下进行填充，而这可以使得我们的代码更加简洁，并且更有效率地利用内存，因为我们不需要另外储存填充操作的结果。一个可以表现这个优势的应用场景就是在结合具有不同长度的特征向量的时候。为了拼接具有不同长度的特征向量，我们一般都先填充输入向量，拼接这个结果然后进行之后的一系列非线性操作等。这是一大类神经网络架构的共同模式。

```python
a = tf.random_uniform([5, 3, 5])
b = tf.random_uniform([5, 1, 6])

# concat a and b and apply nonlinearity
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
d = tf.layers.dense(c, 10, activation=tf.nn.relu)
```

但是这个可以通过广播机制更有效地完成。我们利用事实`f(m(x+y))=f(mx+my)f(m(x+y))=f(mx+my)f(m(x+y))=f(mx+my)`，简化我们的填充操作。因此，我们可以分离地进行这个线性操作，利用广播机制隐式地完成拼接操作。

```python
pa = tf.layers.dense(a, 10, activation=None)
pb = tf.layers.dense(b, 10, activation=None)
d = tf.nn.relu(pa + pb)
```

事实上，这个代码足够通用，并且可以在具有任意形状的张量间应用：

```python
def merge(a, b, units, activation=tf.nn.relu):
    pa = tf.layers.dense(a, units, activation=None)
    pb = tf.layers.dense(b, units, activation=None)
    c = pa + pb
    if activation is not None:
        c = activation(c)
    return c
```

一个更为通用函数形式如上所述：

目前为止，我们讨论了广播机制的优点，但是同样的广播机制也有其缺点，隐式假设几乎总是使得调试变得更加困难，考虑下面的例子：

```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b)
```

你猜这个结果是多少？如果你说是 6，那么你就错了，答案应该是 12。这是因为当两个张量的阶数不匹配的时候，在进行元素间操作之前，TF 将会自动地在更低阶数的张量的第一个维度开始扩展，所以这个加法的结果将会变为`[[2, 3], [3, 4]]`，所以这个`reduce`的结果是12.  

解决这种麻烦的方法就是尽可能地显式使用。我们在需要`reduce`某些张量的时候，显式地指定维度，然后寻找这个 bug 就会变得简单：

```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b, 0)
```

这样，`c`的值就是`[5, 7]`，我们就容易猜到其出错的原因。一个更通用的法则就是总是在`reduce`操作和在使用`tf.squeeze`中指定维度。

## 五、向 TensorFlow 投喂数据

**TensorFlow** 被设计可以在大规模的数据情况下高效地运行。所以你需要记住千万不要“饿着”你的 TF 模型，这样才能得到最好的表现。一般来说，一共有三种方法可以“投喂”你的模型。

### 常数方式（`tf.constant`）

最简单的方式莫过于直接将数据当成常数嵌入你的计算图中，如：

```python
import TensorFlow as tf
import numpy as np

actual_data = np.random.normal(size=[100])
data = tf.constant(actual_data)
```

这个方式非常地高效，但是却不灵活。这个方式存在一个大问题就是为了在其他数据集上复用你的模型，你必须要重写你的计算图，而且你必须同时加载所有数据，并且一直保存在内存里，这意味着这个方式仅仅适用于小数剧集的情况。

### 占位符方式（`tf.placeholder`）

可以通过占位符的方式解决刚才常数投喂网络的问题，如：

```python
import TensorFlow as tf
import numpy as np

data = tf.placeholder(tf.float32)
prediction = tf.square(data) + 1
actual_data = np.random.normal(size=[100])
tf.Session().run(prediction, feed_dict={data: actual_data})
```

占位符操作符返回一个张量，他的值在会话（`session`）中通过人工指定的`feed_dict`参数得到。

### python 操作（`tf.py_func`）

还可以通过利用 python 操作投喂数据：

```python
def py_input_fn():
    actual_data = np.random.normal(size=[100])
    return actual_data

data = tf.py_func(py_input_fn, [], (tf.float32))
```

python 操作允许你将一个常规的 python 函数转换成一个 TF 的操作。

### 利用 TF 的自带数据集 API

最值得推荐的方式就是通过 TF 自带的数据集 API 进行投喂数据，如：

```python
actual_data = np.random.normal(size=[100])
dataset = tf.contrib.data.Dataset.from_tensor_slices(actual_data)
data = dataset.make_one_shot_iterator().get_next()
```

如果你需要从文件中读入数据，你可能需要将文件转化为`TFrecord`格式，这将会使得整个过程更加有效

```python
dataset = tf.contrib.data.Dataset.TFRecordDataset(path_to_data)
```

查看[官方文档](https://www.TensorFlow.org/api_guides/python/reading_data#Reading_from_files)，了解如何将你的数据集转化为`TFrecord`格式。

```python
dataset = ...
dataset = dataset.cache()
if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size * 5)
dataset = dataset.map(parse, num_threads=8)
dataset = dataset.batch(batch_size)
```

在读入了数据之后，我们使用`Dataset.cache()`方法，将其缓存到内存中，以求更高的效率。在训练模式中，我们不断地重复数据集，这使得我们可以多次处理整个数据集。我们也需要打乱数据集得到批量，这个批量将会有不同的样本分布。下一步，我们使用`Dataset.map()`方法，对原始数据进行预处理，将数据转换成一个模型可以识别，利用的格式。然后，我们就通过`Dataset.batch()`，创造样本的批量了。

## 六、利用运算符重载

和 Numpy 一样，TensorFlow 重载了很多 python 中的运算符，使得构建计算图更加地简单，并且使得代码具有可读性。

**切片**操作是重载的诸多运算符中的一个，它可以使得索引张量变得很容易：

```python
z = x[begin:end]  # z = tf.slice(x, [begin], [end-begin])
```

但是在使用它的过程中，你还是需要非常地小心。切片操作非常低效，因此最好避免使用，特别是在切片的数量很大的时候。为了更好地理解这个操作符有多么地低效，我们先观察一个例子。我们想要人工实现一个对矩阵的行进行`reduce`操作的代码：

```python
import TensorFlow as tf
import time

x = tf.random_uniform([500, 10])

z = tf.zeros([10])
for i in range(500):
    z += x[i]

sess = tf.Session()
start = time.time()
sess.run(z)
print("Took %f seconds." % (time.time() - start))
```

在笔者的 MacBook Pro 上，这个代码花费了 2.67 秒！那么耗时的原因是我们调用了切片操作 500 次，这个运行起来超级慢的！一个更好的选择是使用`tf.unstack()`操作去将一个矩阵切成一个向量的列表，而这只需要一次就行！

```python
z = tf.zeros([10])
for x_i in tf.unstack(x):
    z += x_i
```

这个操作花费了 0.18 秒，当然，最正确的方式去实现这个需求是使用`tf.reduce_sum()`操作：

```python
z = tf.reduce_sum(x, axis=0)
```

这个仅仅使用了 0.008 秒，是原始实现的 300 倍！
TensorFlow 除了切片操作，也重载了一系列的数学逻辑运算，如：

```python
z = -x  # z = tf.negative(x)
z = x + y  # z = tf.add(x, y)
z = x - y  # z = tf.subtract(x, y)
z = x * y  # z = tf.mul(x, y)
z = x / y  # z = tf.div(x, y)
z = x // y  # z = tf.floordiv(x, y)
z = x % y  # z = tf.mod(x, y)
z = x ** y  # z = tf.pow(x, y)
z = x @ y  # z = tf.matmul(x, y)
z = x > y  # z = tf.greater(x, y)
z = x >= y  # z = tf.greater_equal(x, y)
z = x < y  # z = tf.less(x, y)
z = x <= y  # z = tf.less_equal(x, y)
z = abs(x)  # z = tf.abs(x)
z = x & y  # z = tf.logical_and(x, y)
z = x | y  # z = tf.logical_or(x, y)
z = x ^ y  # z = tf.logical_xor(x, y)
z = ~x  # z = tf.logical_not(x)
```

你也可以使用这些操作符的增广版本，如 `x += y`和`x **=2`同样是合法的。  
注意到 python 不允许重载`and`，`or`和`not`等关键字。  

TensorFlow 也不允许把张量当成`boolean`类型使用，因为这个很容易出错：

```python
x = tf.constant(1.)
if x:  # 这个将会抛出TypeError错误
    ...
```

如果你想要检查这个张量的值的话，你也可以使用`tf.cond(x,...)`，或者使用`if x is None`去检查这个变量的值。  

有些操作是不支持的，比如说等于判断`==`和不等于判断`!=`运算符，这些在 numpy 中得到了重载，但在 TF 中没有重载。如果需要使用，请使用这些功能的函数版本`tf.equal()`和`tf.not_equal()`。

## 七、理解执行顺序和控制依赖

我们知道，TensorFlow 是属于符号式编程的，它不会直接运行定义了的操作，而是在计算图中创造一个相关的节点，这个节点可以用`Session.run()`进行执行。这个使得 TF 可以在优化过程中决定优化的顺序，并且在运算中剔除一些不需要使用的节点，而这一切都发生在运行中。如果你只是在计算图中使用`tf.Tensors`，你就不需要担心依赖问题，但是你更可能会使用`tf.Variable()`，这个操作使得问题变得更加困难。笔者的建议是如果张量不能满足这个工作需求，那么仅仅使用`Variables`就足够了。这个可能不够直观，我们不妨先观察一个例子：

```python
import TensorFlow as tf

a = tf.constant(1)
b = tf.constant(2)
a = a + b

tf.Session().run(a)
```

计算`a`将会返回 3，就像期望中的一样。注意到我们现在有 3 个张量，两个常数张量和一个储存加法结果的张量。注意到我们不能重写一个张量的值，如果我们想要改变张量的值，我们就必须要创建一个新的张量，就像我们刚才做的那样。

> **小提示：**如果你没有显式地定义一个新的计算图，TF 将会自动地为你构建一个默认的计算图。你可以使用`tf.get_default_graph()`去获得一个计算图的句柄，然后，你就可以查看这个计算图了。比如，可以打印属于这个计算图的所有张量之类的的操作都是可以的。如：

```python
print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
```

不像张量，变量可以更新，所以让我们用变量去实现我们刚才的需求：

```python
a = tf.Variable(1)
b = tf.constant(2)
assign = tf.assign(a, a + b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(assign))
```

同样，我们得到了 3，正如预期一样。注意到`tf.assign()`返回的代表这个赋值操作的张量。目前为止，所有事情都显得很棒，但是让我们观察一个稍微有点复杂的例子吧：

```python
a = tf.Variable(1)
b = tf.constant(2)
c = a + b

assign = tf.assign(a, 5)

sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([assign, c]))
```

注意到，张量`c`并没有一个确定性的值。这个值可能是 3 或者 7，取决于加法和赋值操作谁先运行。  

你应该也注意到了，你在代码中定义操作的顺序是不会影响到在 TF 运行时的执行顺序的，唯一会影响到执行顺序的是**控制依赖**。控制依赖对于张量来说是直接的。每一次你在操作中使用一个张量时，操作将会定义一个对于这个张量来说的隐式的依赖。但是如果你同时也使用了变量，事情就变得更糟糕了，因为变量可以取很多值。  

当处理这些变量时，你可能需要显式地去通过使用`tf.control_dependencies()`去控制依赖，如：

```python
a = tf.Variable(1)
b = tf.constant(2)
c = a + b

with tf.control_dependencies([c]):
    assign = tf.assign(a, 5)

sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([assign, c]))
```

这会确保赋值操作在加法操作之后被调用。
