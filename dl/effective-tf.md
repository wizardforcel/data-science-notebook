# Effective TensorFlow Chapter 1: Tensorflow基础知识

Tensorflow和其他数字计算库（如numpy）之间最明显的区别在于Tensorflow中操作的是符号。这是一个强大的功能，这保证了Tensorflow可以做很多其他库（例如numpy）不能完成的事情（例如自动区分）。这可能也是它更复杂的原因。今天我们来一步步探秘Tensorflow，并为更有效地使用Tensorflow提供了一些指导方针和最佳实践。

我们从一个简单的例子开始，我们要乘以两个随机矩阵。首先我们来看一下在numpy中如何实现：

```bash
import numpy as np
x = np.random.normal(size=[10, 10])
y = np.random.normal(size=[10, 10])
z = np.dot(x, y)
print(z)

```

现在我们使用Tensorflow中执行完全相同的计算：  

```bash
import tensorflow as tf
x = tf.random_normal([10, 10])
y = tf.random_normal([10, 10])
z = tf.matmul(x, y)
sess = tf.Session()
z_val = sess.run(z)
print(z_val)

```

与立即执行计算并将结果复制给输出变量z的numpy不同，tensorflow只给我们一个可以操作的张量类型。如果我们尝试直接打印z的值，我们得到这样的东西：  

```py
Tensor("MatMul:0", shape=(10, 10), dtype=float32)
```

由于两个输入都是已经定义的类型，tensorFlow能够推断张量的符号及其类型。为了计算张量的值，我们需要创建一个会话并使用Session.run方法进行评估。

要了解如此强大的符号计算到底是什么，我们可以看看另一个例子。假设我们有一个曲线的样本（例如f（x）= 5x ^ 2 + 3），并且我们要估计f（x）在不知道它的参数的前提下。我们定义参数函数为g（x，w）= w0 x ^ 2 + w1 x + w2，它是输入x和潜在参数w的函数，我们的目标是找到潜在参数，使得g（x， w）≈f（x）。这可以通过最小化损失函数来完成：L（w）=（f（x）-g（x，w））^ 2。虽然这问题有一个简单的封闭式的解决方案，但是我们选择使用一种更为通用的方法，可以应用于任何可以区分的任务，那就是使用随机梯度下降。我们在一组采样点上简单地计算相对于w的L（w）的平均梯度，并沿相反方向移动。

以下是在Tensorflow中如何完成：  

```py
import numpy as np
import tensorflow as tf
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

这只是Tensorflow可以做的冰山一角。许多问题，如优化具有数百万个参数的大型神经网络，都可以在Tensorflow中使用短短的几行代码高效地实现。而且Tensorflow可以跨多个设备和线程进行扩展，并支持各种平台。

**理解静态形状和动态形状的区别：**

Tensorflow中的张量在图形构造期间具有静态的形状属性。例如，我们可以定义一个形状的张量\[None，128\]：  

```bash
import tensorflow as tf
a = tf.placeholder([None, 128])
```

这意味着第一个维度可以是任意大小的，并且将在Session.run期间随机确定。Tensorflow有一个非常简单的API来展示静态形状：  

```bash
static_shape = a.get_shape().as_list()  # returns [None, 128]
```

为了获得张量的动态形状，你可以调用tf.shape op，它将返回一个表示给定形状的张量：

```bash
dynamic_shape = tf.shape(a)
```

我们可以使用Tensor.set_shape（）方法设置张量的静态形状：  

`a.set_shape([32, 128])`

实际上使用tf.reshape（）操作更为安全：  

```py
a =  tf.reshape(a, [32, 128])
```

这里有一个函数可以方便地返回静态形状，当静态可用而动态不可用的时候。

```py
def get_shape(tensor):
  static_shape = tensor.get_shape().as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

```

现在想象一下，如果我们要将三维的张量转换成二维的张量。在TensorFlow中我们可以使用get_shape（）函数：  

```py
b = placeholder([None, 10, 32])
shape = get_shape(tensor)
b = tf.reshape(b, [shape[0], shape[1] * shape[2]])

```

请注意，无论是否静态指定形状，都可以这样做。

实际上，我们可以写一个通用的重塑功能来如何维度之间的转换：  

```bash
import tensorflow as tf
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

然后转化为二维就变得非常容易了：  

```bash
b = placeholder([None, 10, 32])
b = tf.reshape(b, [0, [1, 2]])

```

**广播机制（****broadcasting****）的好与坏：**

Tensorflow同样支持[广播机制](http://blog.csdn.net/lanchunhui/article/details/50158975)。当要执行加法和乘法运算时，你需要确保操作数的形状匹配，例如，你不能将形状\[3，2\]的张量添加到形状的张量\[3,4\]。但有一个特殊情况，那就是当你有一个单一的维度。Tensorflow隐含地功能可以将张量自动匹配另一个操作数的形状。例如：  

```py
import tensorflow as tf
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[1.], [2.]])
# c = a + tf.tile(a, [1, 2])
c = a + b 

```

广播允许我们执行隐藏的功能，这使代码更简单，并且提高了内存的使用效率，因为我们不需要再使用其他的操作。为了连接不同长度的特征，我们通常平铺式的输入张量。这是各种神经网络架构的最常见模式：  

```py
a = tf.random_uniform([5, 3, 5])
b = tf.random_uniform([5, 1, 6])
# concat a and b and apply nonlinearity
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
d = tf.layers.dense(c, 10, activation=tf.nn.relu)

```

这可以通过广播机制更有效地完成。我们使用f（m（x + y））等于f（mx + my）的事实。所以我们可以分别进行线性运算，并使用广播进行隐式级联：

pa = tf.layers.dense(a, 10, activation=None)  

```py
pb = tf.layers.dense(b, 10, activation=None)
d = tf.nn.relu(pa + pb)

```

实际上，这段代码很普遍，只要在张量之间进行广播就可以应用于任意形状的张量：  

```bash
def tile_concat_dense(a, b, units, activation=tf.nn.relu):
    pa = tf.layers.dense(a, units, activation=None)
    pb = tf.layers.dense(b, units, activation=None)
    c = pa + pb
    if activation is not None:
        c = activation(c)
    return c
```

到目前为止，我们讨论了广播的好的部分。但是你可能会问什么坏的部分？隐含的假设总是使调试更加困难，请考虑以下示例：  

```py
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b)

```

你认为C的数值是多少如果你猜到6，那是错的。这是因为当两个张量的等级不匹配时，Tensorflow会在元素操作之前自动扩展具有较低等级的张量，因此加法的结果将是\[\[2,3\]， \[3，4\]\]。

如果我们指定了我们想要减少的维度，避免这个错误就变得很容易了：  

```py
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b, 0)

```

这里c的值将是\[5,7\]。

**使用****Python****实现原型内核和高级可视化的操作：**

为了提高效率，Tensorflow中的操作内核完全是用C ++编写，但是在C ++中编写Tensorflow内核可能会相当痛苦。。使用tf.py_func（），你可以将任何python代码转换为Tensorflow操作。

例如，这是python如何在Tensorflow中实现一个简单的ReLU非线性内核：

```py
import numpy as np
import tensorflow as tf
import uuid
def relu(inputs):
    # Define the op in python
    def _relu(x):
        return np.maximum(x, 0.)
    # Define the op's gradient in python
    def _relu_grad(x):
        return np.float32(x > 0)
    # An adapter that defines a gradient op compatible with Tensorflow
    def _relu_grad_op(op, grad):
        x = op.inputs[0]
        x_grad = grad * tf.py_func(_relu_grad, [x], tf.float32)
        return x_grad
    # Register the gradient with a unique id
    grad_name = "MyReluGrad_" + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_relu_grad_op)
    # Override the gradient of the custom op
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        output = tf.py_func(_relu, [inputs], tf.float32)
    return output

```

要验证梯度是否正确，你可以使用Tensorflow的梯度检查器：  

```bash
x = tf.random_normal([10])
y = relu(x * x)
with tf.Session():
    diff = tf.test.compute_gradient_error(x, [10], y, [10])
    print(diff)

```

compute\_gradient\_error（）是以数字的方式计算梯度，并返回与渐变的差异，因为我们想要的是一个很小的差异。

请注意，此实现效率非常低，只对原型设计有用，因为python代码不可并行化，不能在GPU上运行。

在实践中，我们通常使用python ops在Tensorboard上进行可视化。试想一下你正在构建图像分类模型，并希望在训练期间可视化你的模型预测。Tensorflow允许使用函数tf.summary.image（）进行可视化：  

```py
image = tf.placeholder(tf.float32)
tf.summary.image("image", image)

```

但这只能显示输入图像，为了可视化预测，你必须找到一种方法来添加对图像的注释，这对于现有操作几乎是不可能的。一个更简单的方法是在python中进行绘图，并将其包装在一个python 方法中：  

```py
import io
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
def visualize_labeled_images(images, labels, max_outputs=3, name='image'):
    def _visualize_image(image, label):
        # Do the actual drawing in python
        fig = plt.figure(figsize=(3, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.imshow(image[::-1,...])
        ax.text(0, 0, str(label), 
          horizontalalignment='left', 
          verticalalignment='top')
        fig.canvas.draw()
        # Write the plot as a memory file.
        buf = io.BytesIO()
        data = fig.savefig(buf, format='png')
        buf.seek(0)      
        # Read the image and convert to numpy array
        img = PIL.Image.open(buf)
        return np.array(img.getdata()).reshape(img.size[0], img.size[1], -1)
    def _visualize_images(images, labels):
        # Only display the given number of examples in the batch
        outputs = []
        for i in range(max_outputs):
            output = _visualize_image(images[i], labels[i])
            outputs.append(output)
        return np.array(outputs, dtype=np.uint8)
    # Run the python op.
    figs = tf.py_func(_visualize_images, [images, labels], tf.uint8)
    return tf.summary.image(name, figs)

```

请注意，由于概要通常只能在一段时间内进行评估（不是每步），因此实施中可以使用该实现，而不用担心效率。

本文由北邮[@爱可可-爱生活老师](http://weibo.com/fly51fly?from=feed&loc=at&nick=%E7%88%B1%E5%8F%AF%E5%8F%AF-%E7%88%B1%E7%94%9F%E6%B4%BB)推荐，@阿里云云栖社区组织翻译。

文章原标题《Effective Tensorflow - Guides and best practices for effective use of Tensorflow》

作者：**Vahid Kazemi** 作者是**google的**件工程师，CS中的博士学位。从事机器学习，NLP和计算机视觉工作。

译者：袁虎 审阅：

文章为简译，更为详细的内容，请查看[原文](https://github.com/vahidk/EffectiveTensorflow?spm=a2c4e.11153940.blogcont159607.11.57c5884dyUjY7u "原文")

# Effective TensorFlow Chapter 2: 理解静态和动态的Tensor类型的形状

**本文翻译自： [《Understanding static and dynamic shapes》](http://usyiyi.cn/translate/effective-tf/1.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

译者：[FesianXu](http://my.csdn.net/LoseInVain)

* * *

在**TensorFlow**中，`tensor`有一个在图构建过程中就被决定的**静态形状属性**， 这个静态形状可以是**没有明确加以说明的(underspecified)**，比如，我们可以定一个具有形状**\[None, 128\]**大小的`tensor`。

```python
import tensorflow as tf
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

tensor的静态形状可以通过方法`Tensor_name.set_shape()`设定，如：

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

现在，如果我们需要将一个三阶的`tensor`转变为2阶的`tensor`，通过折叠（collapse）第二维和第三维成一个维度，我们可以通过我们刚才定义的`get_shape()`方法进行，如：

```python
b = tf.placeholder(tf.float32, [None, 10, 32])
shape = get_shape(b)
b = tf.reshape(b, [shape[0], shape[1] * shape[2]])
```

注意到无论这个`tensor`的形状是静态指定的还是动态指定的，这个代码都是有效的。事实上，我们可以写出一个通用的reshape函数，用于折叠任意在列表中的维度（any list of dimensions）:

```python
import tensorflow as tf
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

# Effective TensorFlow Chapter 3: 理解变量域Scope和何时应该使用它

**本文翻译自： [《Scopes and when to use them》](http://usyiyi.cn/translate/effective-tf/1.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

译者：[FesianXu](http://my.csdn.net/LoseInVain)

* * *

在TensorFlow中，变量(**Variables**)和张量(**tensors**)有一个名字属性，用于作为他们在图中的标识。如果你在创造变量或者张量的时候，不给他们显式地指定一个名字，那么TF将会自动地，隐式地给他们分配名字，如：

```python
a = tf.constant(1)
print(a.name)  # prints "Const:0"

b = tf.Variable(1)
print(b.name)  # prints "Variable:0"
```

你也可以在定义的时候，通过显式地给变量或者张量命名，这样将会重写（**overwrite**）他们的默认名，如：

```python
a = tf.constant(1, name="a")
print(a.name)  # prints "b:0"

b = tf.Variable(1, name="b")
print(b.name)  # prints "b:0"
```

TF引进了两个不同的上下文管理器，用于更改张量或者变量的名字，第一个就是`tf.name_scope`，如：

```python
with tf.name_scope("scope"):
  a = tf.constant(1, name="a")
  print(a.name)  # prints "scope/a:0"

  b = tf.Variable(1, name="b")
  print(b.name)  # prints "scope/b:0"

  c = tf.get_variable(name="c", shape=[])
  print(c.name)  # prints "c:0"
```

我们注意到，在TF中，我们有两种方式去定义一个新的变量，通过`tf.Variable()`或者调用`tf.get_variable()`。在调用`tf.get_variable()`的时候，给予一个新的名字，将会创建一个新的变量，但是如果这个名字并不是一个新的名字，而是已经存在过这个变量空间（variable scope）中的，那么就会抛出一个ValueError异常，意味着重复声明一个变量是不被允许的。

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

这个语法可能看起来并不是特别的清晰明了。特别是，如果你在模型中想要实现一大堆的变量共享，你需要追踪各个变量，比如说什么时候定义新的变量，什么时候要复用他们，这些将会变得特别麻烦而且容易出错，因此TF提供了TF模版（**TensorFlow templates**）自动解决变量共享的问题：

```python
conv3x32 = tf.make_template("conv3x32", lambda x: tf.layers.conv2d(x, 32, 3))
features1 = conv3x32(image1)
features2 = conv3x32(image2)  # Will reuse the convolution weights.
```

你可以将任何函数都转换为TF模版。当第一次调用这个模版的时候，在这个函数内声明的变量将会被定义，同时在接下来的连续调用中，这些变量都将自动地复用。

# Effective TensorFlow Chapter 4: TensorFlow中的广播Broadcast机制

**本文翻译自： [《Broadcasting the good and the ugly》](http://usyiyi.cn/translate/effective-tf/4.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

译者：[FesianXu](http://my.csdn.net/LoseInVain)

* * *

TensorFlow支持广播机制（**Broadcast**），可以广播元素间操作(**elementwise operations**)。正常情况下，当你想要进行一些操作如加法，乘法时，你需要确保操作数的形状是相匹配的，如：你不能将一个具有形状\[3, 2\]的张量和一个具有\[3,4\]形状的张量相加。但是，这里有一个特殊情况，那就是当你的其中一个操作数是一个具有单独维度(**singular dimension**)的张量的时候，TF会隐式地在它的单独维度方向填满(**tile**)，以确保和另一个操作数的形状相匹配。所以，对一个\[3,2\]的张量和一个\[3,1\]的张量相加在TF中是合法的。（**译者：这个机制继承自numpy的广播功能。其中所谓的单独维度就是一个维度为1，或者那个维度缺失，具体可参考[numpy broadcast](https://www.cnblogs.com/yangmang/p/7125458.html)**）。

```python
import tensorflow as tf

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[1.], [2.]])
# c = a + tf.tile(b, [1, 2])
c = a + b
```

广播机制允许我们在隐式情况下进行填充（**tile**），而这可以使得我们的代码更加简洁，并且更有效率地利用内存，因为我们不需要另外储存填充操作的结果。一个可以表现这个优势的应用场景就是在结合具有不同长度的特征向量的时候。为了拼接具有不同长度的特征向量，我们一般都先填充输入向量，拼接这个结果然后进行之后的一系列非线性操作等。这是一大类神经网络架构的共同套路(**common pattern**)

```python
a = tf.random_uniform([5, 3, 5])
b = tf.random_uniform([5, 1, 6])

# concat a and b and apply nonlinearity
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
d = tf.layers.dense(c, 10, activation=tf.nn.relu)
```

但是这个可以通过广播机制更有效地完成。我们利用事实f(m(x+y))=f(mx+my)f(m(x+y))=f(mx+my)，简化我们的填充操作。因此，我们可以分离地进行这个线性操作，利用广播机制隐式地完成拼接操作。

```python
pa = tf.layers.dense(a, 10, activation=None)
pb = tf.layers.dense(b, 10, activation=None)
d = tf.nn.relu(pa + pb)
```

事实上，这个代码足够通用，并且可以在具有抽象形状(**arbitrary shape**)的张量间应用：

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

你猜这个结果是多少？如果你说是6，那么你就错了，答案应该是12.这是因为当两个张量的阶数不匹配的时候，在进行元素间操作之前，TF将会自动地在更低阶数的张量的第一个维度开始扩展，所以这个加法的结果将会变为\[\[2, 3\], \[3, 4\]\]，所以这个reduce的结果是12.  
（**译者：答案详解如下，第一个张量的shape为\[2, 1\]，第二个张量的shape为\[2,\]。因为从较低阶数张量的第一个维度开始扩展，所以应该将第二个张量扩展为shape=\[2,2\]，也就是值为\[\[1,2\], \[1,2\]\]。第一个张量将会变成shape=\[2,2\]，其值为\[\[1, 1\], \[2, 2\]\]。**）  
解决这种麻烦的方法就是尽可能地显示使用。我们在需要reduce某些张量的时候，显式地指定维度，然后寻找这个bug就会变得简单：

```python
a = tf.constant([[1.], [2.]])
b = tf.constant([1., 2.])
c = tf.reduce_sum(a + b, 0)
```

这样，c的值就是\[5, 7\]，我们就容易猜到其出错的原因。一个更通用的法则就是总是在reduce操作和在使用`tf.squeeze`中指定维度。

# Effective TensorFlow Chapter 5: 在TensorFlow中，给模型喂数据(feed data)

**本文翻译自： [《Feeding data to TensorFlow》](http://usyiyi.cn/translate/effective-tf/5.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

译者：[FesianXu](http://my.csdn.net/LoseInVain)

* * *

**TensorFlow**被设计可以在大规模的数据情况下高效地运行。所以你需要记住千万不要“饿着”你的TF模型，这样才能得到最好的表现。一般来说，一共有三种方法可以“喂养”（**feed**）你的模型。

## 常数方式（**Constants**）

最简单的方式莫过于直接将数据当成常数嵌入你的计算图中，如：

```python
import tensorflow as tf
import numpy as np

actual_data = np.random.normal(size=[100])
data = tf.constant(actual_data)
```

这个方式非常地高效，但是却不灵活。这个方式存在一个大问题就是为了在其他数据集上复用你的模型，你必须要重写你的计算图，而且你必须同时加载所有数据，并且一直保存在内存里，这意味着这个方式仅仅适用于小数剧集的情况。

## 占位符方式（**Placeholders**）

可以通过占位符(**placeholder**)的方式解决刚才常数喂养网络的问题，如：

```python
import tensorflow as tf
import numpy as np

data = tf.placeholder(tf.float32)
prediction = tf.square(data) + 1
actual_data = np.random.normal(size=[100])
tf.Session().run(prediction, feed_dict={data: actual_data})
```

占位符操作符返回一个张量，他的值在会话(**session**)中通过人工指定的`feed_dict`参数得到(**fetch**)。（**译者：也就是说占位符其实只是占据了数据喂养的位置而已，而不是真正的数据，所以在训练过程中，如果真正需要使用这个数据，就必须要指定合法的feed_dict，否则将会报错。**）

## 通过python的操作（**python ops**）

还可以通过利用python ops喂养数据：

```python
def py_input_fn():
    actual_data = np.random.normal(size=[100])
    return actual_data

data = tf.py_func(py_input_fn, [], (tf.float32))
```

python ops允许你将一个常规的python函数转换成一个TF的操作。（**译者：这种方法不常用。**）

## 利用TF的自带数据集API（**Dataset API**）

最值得推荐的方式就是通过TF自带的数据集API进行喂养数据，如：

```python
actual_data = np.random.normal(size=[100])
dataset = tf.contrib.data.Dataset.from_tensor_slices(actual_data)
data = dataset.make_one_shot_iterator().get_next()
```

如果你需要从文件中读入数据，你可能需要将文件转化为`TFrecord`格式，这将会使得整个过程更加有效（**译者：同时，可以利用TF中的队列机制和多线程机制，实现无阻塞的训练。**）

```python
dataset = tf.contrib.data.Dataset.TFRecordDataset(path_to_data)
```

查看[官方文档](https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files)，了解如何将你的数据集转化为`TFrecord`格式。（**译者：我即将推出关于TFrecord的博文，有需要的朋友敬请关注。**）

```python
dataset = ...
dataset = dataset.cache()
if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size * 5)
dataset = dataset.map(parse, num_threads=8)
dataset = dataset.batch(batch_size)
```

在读入了数据之后，我们使用`Dataset.cache()`方法，将其缓存到内存中，以求更高的效率。在训练模式中，我们不断地重复数据集，这使得我们可以多次处理整个数据集。我们也需要打乱（**shuffle**）数据集得到batch，这个batch将会有不同的样本分布。下一步，我们使用`Dataset.map()`方法，对原始的数据（**raw records**）进行预处理，将数据转换成一个模型可以识别，利用的格式。（**译者：map参考MapDeduce和python自带的高阶函数map**）然后，我们就通过`Dataset.batch()`，创造样本的batch了。

# Effective TensorFlow Chapter 6: 在TensorFlow中的运算符重载

**本文翻译自： [《Take advantage of the overloaded operators》](http://usyiyi.cn/translate/effective-tf/6.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

译者：[FesianXu](http://my.csdn.net/LoseInVain)

* * *

和Numpy一样，TensorFlow重载了很多python中的运算符，使得构建计算图更加地简单，并且使得代码具有可读性。  
**切片（slice）**操作是重载的诸多运算符中的一个，它可以使得索引张量变得很容易：

```python
z = x[begin:end]  # z = tf.slice(x, [begin], [end-begin])
```

但是在使用它的过程中，你还是需要非常地小心。切片操作非常低效，因此最好避免使用，特别是在切片的数量很大的时候。为了更好地理解这个操作符有多么地低效，我们先观察一个例子。我们想要人工实现一个对矩阵的行进行reduce操作的代码：

```python
import tensorflow as tf
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

在笔者的MacBook Pro上，这个代码花费了2.67秒！那么耗时的原因是我们调用了切片操作500次，这个运行起来超级慢的！一个更好的选择是使用`tf.unstack()`操作去将一个矩阵切成一个向量的列表，而这只需要一次就行！

```python
z = tf.zeros([10])
for x_i in tf.unstack(x):
    z += x_i
```

这个操作花费了0.18秒，当然，最正确的方式去实现这个需求是使用`tf.reduce_sum()`操作：

```python
z = tf.reduce_sum(x, axis=0)
```

这个仅仅使用了0.008秒，是原始实现的300倍！  
TensorFlow除了切片操作，也重载了一系列的数学逻辑运算，如：

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
注意到python不允许重载`and`,`or`和`not`等关键字。  
TensorFlow也不允许把张量当成`boolean`类型使用，因为这个很容易出错：

```python
x = tf.constant(1.)
if x:  # 这个将会抛出TypeError错误
    ...
```

如果你想要检查这个张量的值的话，你也可以使用`tf.cond(x,...)`，或者使用`if x is None`去检查这个变量的值。  
有些操作是不支持的，比如说等于判断`==`和不等于判断`!=`运算符，这些在numpy中得到了重载，但在TF中没有重载。如果需要使用，请使用这些功能的函数版本`tf.equal()`和`tf.not_equal()`。

# Effective TensorFlow Chapter 7: TensorFlow中的执行顺序和控制依赖

**本文翻译自： [《Understanding order of execution and control dependencies》](http://usyiyi.cn/translate/effective-tf/7.html)， 如有侵权请联系删除，仅限于学术交流，请勿商用。如有谬误，请联系指出。**

译者：[FesianXu](http://my.csdn.net/LoseInVain)

* * *

我们知道，TensorFlow是属于符号式编程的，它不会直接运行定义了的操作，而是在计算图中创造一个相关的节点，这个节点可以用`Session.run()`进行执行。这个使得TF可以在优化过程中(**do optimization**)决定优化的顺序(**the optimal order**)，并且在运算中剔除一些不需要使用的节点，而这一切都发生在运行中(**run time**)。如果你只是在计算图中使用`tf.Tensors`，你就不需要担心依赖问题（**dependencies**），但是你更可能会使用`tf.Variable()`，这个操作使得问题变得更加困难。笔者的建议是如果张量不能满足这个工作需求，那么仅仅使用`Variables`就足够了。这个可能不够直观，我们不妨先观察一个例子：

```python
import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
a = a + b

tf.Session().run(a)
```

计算`a`将会返回3，就像期望中的一样。注意到我们现在有3个张量，两个常数张量和一个储存加法结果的张量。注意到我们不能重写一个张量的值（**译者：这个很重要，张量在TF中表示操作单元，是一个操作而不是一个值，不能进行赋值操作等。**），如果我们想要改变张量的值，我们就必须要创建一个新的张量，就像我们刚才做的那样。

> **小提示：**如果你没有显式地定义一个新的计算图，TF将会自动地为你构建一个默认的计算图。你可以使用`tf.get_default_graph()`去获得一个计算图的句柄（handle），然后，你就可以查看这个计算图了。比如，可以打印属于这个计算图的所有张量之类的的操作都是可以的。如：

```python
print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
```

不像张量，变量Variables可以更新，所以让我们用变量去实现我们刚才的需求：

```python
a = tf.Variable(1)
b = tf.constant(2)
assign = tf.assign(a, a + b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(assign))
```

同样，我们得到了3，正如预期一样。注意到`tf.assign()`返回的代表这个赋值操作的张量。目前为止，所有事情都显得很棒，但是让我们观察一个稍微有点复杂的例子吧：

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

（**译者：这个会输出\[7, 5\]**）

注意到，张量`c`并没有一个确定性的值。这个值可能是3或者7，取决于加法和赋值操作谁先运行。  
你应该也注意到了，你在代码中定义操作(**ops**)的顺序是不会影响到在TF运行时的执行顺序的，唯一会影响到执行顺序的是**控制依赖**。控制依赖对于张量来说是直接的。每一次你在操作中使用一个张量时，操作将会定义一个对于这个张量来说的隐式的依赖。但是如果你同时也使用了变量，事情就变得更糟糕了，因为变量可以取很多值。  
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

（**译者：这个会输出\[5, 3\]**）

这会确保赋值操作在加法操作之后被调用。

**译者：**  
这里贴出`tf.control_dependencies()`的API文档，希望有所帮助：

**tf.control\_dependencies(control\_inputs)**

**control_inputs**：一个操作或者张量的列表，这个列表里面的东西必须在运行定义在下文中的操作执行之前执行。当然也可以为None，这样会消除控制依赖的作用。

