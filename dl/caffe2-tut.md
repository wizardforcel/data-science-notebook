# Caffe2 Tutorials[0]

本系列教程包括9个小节，对应Caffe2官网的前9个教程，第10个教程讲的是在安卓下用SqueezeNet进行物体检测，此处不再翻译。另外由于栏主不关注RNN和LSTM，所以栏主不对剩下两个教程翻译。有志翻译的同学可向本专栏投稿。

![](https://upload-images.jianshu.io/upload_images/5064138-ae7d340f4cb9aa2c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

1.  \[Caffe2 Tutorials Overview\] [翻译](https://www.jianshu.com/p/b2273e7d1d37) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorials.html)
2.  \[Intro Tutorial\] [翻译](https://www.jianshu.com/p/f1092b1e6822) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/intro-tutorial.html)
3.  \[Models and Datasets\] [翻译](https://www.jianshu.com/p/aa9c9aba0829) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorial-models-and-datasets.html)
4.  \[Basics of Caffe2 - Workspaces, Operators, and Nets\] [翻译](https://www.jianshu.com/p/8a81d40bbf6c) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorial-basics-of-caffe2.html)
5.  \[Toy Regression\] [翻译](https://www.jianshu.com/p/b2c61c7716b5) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorial-toy-regression.html)
6.  \[Image Pre-Processing\] [翻译](https://www.jianshu.com/p/e15951bfe8a3) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorial-image-pre-processing.html)
7.  \[Loading Pre-Trained Models\] [翻译](https://www.jianshu.com/p/c8539e2ec01b) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorial-loading-pre-trained-models.html)
8.  \[MNIST - Create a CNN from Scratch\] [翻译](https://www.jianshu.com/p/90fb77649b51) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorial-MNIST.html)
9.  \[Create Your Own Dataset\] [翻译](https://www.jianshu.com/p/14e04d7b46ed) & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorial-create-your-own-dataset.html)
10.  \[AI Camera Demo and Tutorial\] 暂无 & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/AI-Camera-demo-android.html)
11.  \[RNNs and LSTM Networks\] 暂无 & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/RNNs-and-LSTM-networks.html)
12.  \[Synchronous SGD\] 暂无 & [原文](https://link.jianshu.com?t=https://caffe2.ai/docs/SynchronousSGD.html)

# Caffe2 用户手册概览（Caffe2 Tutorials Overview）[1]

在开始之前，我们很感激你对Caffe2感兴趣，希望Caffe2在你的机器学习作品中是一个高性能的框架。Caffe2致力于模块化，促进深度学习想法和原型的实现。

# 选择你的学习路线

1\. [使用一个现成的预训练模型（容易）](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorials.html#null__new-to-deep-learning)  

+ 2\. [编写自己的神经网络（中等）](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorials.html#null__creating-a-convolutional-neural-network-from-scratch)  

+ 3\. [移动应用。做一个应用深度学习技术的移动端APP（高级）](https://link.jianshu.com?t=https://caffe2.ai/docs/AI-Camera-demo-android)  

+ 选择1，点击链接，有几个使用预训练模型的例子，我们将会展示如何在几分钟内跑起demo  

+ 选择2，你需要一些深度学习的背景知识。后面会给出一些资料的链接。  

+ 选择3，你将会看到如何在Android或者IOS上运行图像分类的APP。这是完全即插即用的，不过你需要了解Caffe2的C++接口。

# IPython Notebook

在`/caffe/python/examples`目录下有几个程序示例，可以帮助你了解如何使用Caffe2  

+ [char_rnn.py](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/char_rnn.py)：生成一个递归神经网络，对你输入的文本进行抽样，然后随机生成一个类似风格的文本。  

+ [lmdb\_create\_example.py](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/lmdb_create_example.py)：生成一个图片和标签的lmdb的数据库，你可以把这个作为框架写自己的数据读入接口  

+ [resnet50_trainer.py](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py)：多GPU并行训练Resnet-50。可以用来在imagenet上训练。  

+ [seq2seq.py](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/seq2seq.py)：创建一个特殊的能处理文本行的RNN，比如翻译  

+ [seq2seq_util.py](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/seq2seq_util.py)：序列到序列的有用函数

# New to Caffe2

#### [Basics of Caffe2 - Workspaces, Operators, and Nets](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Basics.ipynb)

Caffe2 包含三个概念：  

+  Workspaces  

+  Operators  

+  Nets

#### [Toy Regression - Plotting Lines & Random Data](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Toy_Regression.ipynb)

这个教程主要展示了如何使用Caffe2进行回归  

+  生成随机样本数据  

+  创建网络  

+  自动训练网络  

+  查看梯度下降结果和训练过程中参数的变化

#### [Image Pre-Processing Pipeline](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Image_Pre-Processing_Pipeline.ipynb)

这个例子主要展示了如何进行数据预处理使之适合预训练的模型。  

+  调整  

+  缩放  

+  HWC到CHW的变换（译者注：缩写应该是channel，height，width）  

+  RGB到BGR的变换  

+  图像预处理（译者注：包括减均值，归一化等等）

# Creating a Convolutional Neural Network from Scratch

#### [MNIST - Handwriting Recognition](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/MNIST.ipynb)

这个教程创建一个小小的CNN来识别手写字符。

#### [Create Your Own Dataset](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/create_your_own_dataset.ipynb)

这个教程告诉你如何导入和修改数据使之能在Caffe2中使用。教程使用的是Iris数据集

# Tour of Caffe Components

##### C++ implementation

gpu.h: needs documentation  

+ db.h: needs documentation

##### Python implementation

TensorProtosDBInput: needs documentation

##### Writing Your Own Operators

自定义Operators参考如下教程  

+ [Guide for creating your own operators](https://link.jianshu.com?t=https://caffe2.ai/docs/custom-operators)

##### Tutorials Installation

如果你需要跑起手册里面的例子，你需要安装如下依赖包

```
sudo pip install flask graphviz hypothesis jupyter leveldb lmdb matplotlib pydot pyyaml requests scikit-image scipy tornado zeromq

```

结语：  
转载请注明出处：[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

# Caffe2 手册（Intro Tutorial）[2]

# Caffe2的相关概念

接下来你可以学到更多Caffe2中主要的概念，这些概念对理解和开发Caffe2相当重要。

### Blobs and Workspace，Tensors

Caffe2中，数据是用blobs储存的。Blob只是内存中的一个数据块。大多数Blobs包含一个张量（tensor），可以理解为多维矩阵，在Python中，他们被转换为numpy 矩阵。  

+ `Workspace` 保存着所有的`Blobs`。下面的例子展示了如何向`Workspace`中传递`Blobs`和取出他们。`Workspace`在你开始使用他们时，才进行初始化。

```
# Create random tensor of three dimensions
x = np.random.rand(4, 3, 2)
print(x)
print(x.shape)
workspace.FeedBlob("my_x", x)
x2 = workspace.FetchBlob("my_x")
print(x2)

```

### Nets and Operators

Caffe2中最基本的对象是`net`，`net`可以说是一系列`Operators`的集合，每个`Operator`根据输入的`blob`输出一个或者多个`blob`。  

+ 下面我们将会创建一个超级简单的模型。他拥有如下部件：

+   一个全连接层
+   一个`Sigmoid`激活函数和一个`Softmax`函数
+   一个交叉损失  
    
+ 直接构建网络是很厌烦的，所以最好使用Python接口的模型助手来构建网络。我们只需简单的调用`CNNModelHelper`，他就会帮我们创建两个想联系的网络。
+   一个用于初始化参数（`ref.init_net`）
+   一个用于实际训练（`ref.init_net`）

```
# Create the input data
data = np.random.rand(16, 100).astype(np.float32)
# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)
workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)
# Create model using a model helper
m = cnn.CNNModelHelper(name="my first net")
fc_1 = m.FC("data", "fc1", dim_in=100, dim_out=10)
pred = m.Sigmoid(fc_1, "pred")
[softmax, loss] = m.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

```

上面的代码中，我们首先在内存中创建了输入数据和标签，实际使用中，往往从`database`等载体中读入数据。可以看到输入数据和标签的第一维度是`16`，这是因为输入的最小`batch`最小是`16`。Caffe2中很多`Operator`都能直接通过`CNNModelHelper`来进行，并且能够一次处理一个`batch`。[CNNModelHelper’s Operator List](https://link.jianshu.com?t=https://caffe2.ai/docs/workspace.html#cnnmodelhelper)中有更详细的解析。  

+ 第二，我们通过一些操作创建了一个模型。比如`FC`，`Sigmoid`，`SoftmaxWithLoss`。**注意**：这个时候，这些操作并没有真正执行，他们仅仅是对模型进行了定义。  

+ 模型助手创建了两个网络：`m.param_init_net`，这个网络将仅仅被执行一次。他将会初始化参数`blob`,例如全连接层的权重。真正的训练是通过执行`m.net`来是现实的。这是自动发生的。  

+ 网络的定义保存在一个`protobuf`结构体中。你可以很容易的通过调用`net.proto`来查看它。

```
print(str(m.net.Proto()))

```

输出如下：

```
name: "my first net"
op {
  input: "data"
  input: "fc1_w"
  input: "fc1_b"
  output: "fc1"
  name: ""
  type: "FC"
}
op {
  input: "fc1"
  output: "pred"
  name: ""
  type: "Sigmoid"
}
op {
  input: "pred"
  input: "label"
  output: "softmax"
  output: "loss"
  name: ""
  type: "SoftmaxWithLoss"
}
external_input: "data"
external_input: "fc1_w"
external_input: "fc1_b"
external_input: "label"

```

同时，你也可以查看参数初始化网络：

```
print(str(m.param_init_net.Proto()))

```

这就是Caffe2的API：使用Python接口方便快速的构建网络并训练你的模型，Python接口将这些网络通过序列化的`protobuf`传递给C++接口，然后C++接口全力的执行。

### Executing

现在我们可以开始训练我们的模型。  

+ 首先，我们先跑一次参数初始化网络。

```
workspace.RunNetOnce(m.param_init_net)

```

这个操作将会把`param_init_net`的`protobuf`传递给C++代码进行执行。  

+ **然后我们真正的创建网络**：

```
workspace.CreateNet(m.net)

```

一旦创建好网络，我们就可以高效的跑起来：

```
# Run 100 x 10 iterations 跑100*10次迭代
for j in range(0, 100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)
    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)
    workspace.RunNet(m.name, 10)   # run for 10 times 跑十次

```

这里要注意的是我们怎样在`RunNet()`函数中使用网络的名字。并且在这里，由于网络已经在`workspace`中创建，所以我们不需要再传递网络的定义。执行完后，你可以查看存在输出`blob`中的结果。

```
print(workspace.FetchBlob("softmax"))
print(workspace.FetchBlob("loss"))

```

### Backward pass

上面的网络中，仅仅包含了网络的前向传播，因此它是学习不到任何东西的。后向传播对每一个前向传播进行`gradient operator`。如果你想自己尝试这样的操作，那么你可以进行以下操作并检查结果。  

+ **在`RunNetOnce()`前**，插入下面操作：

```
m.AddGradientOperators([loss])

```

然后测试`protobuf`的输出：

```
print(str(m.net.Proto()))

```

以上就是大体的使用教程  
**译者注**：  
训练过程可以总结为以下步骤：

```
# Create model using a model helper
m = cnn.CNNModelHelper(name="my first net")
fc_1 = m.FC("data", "fc1", dim_in=100, dim_out=10)
pred = m.Sigmoid(fc_1, "pred")
[softmax, loss] = m.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])
m.AddGradientOperators([loss]) #注意这一行代码
workspace.RunNetOnce(m.param_init_net)
workspace.CreateNet(m.net)
# Run 100 x 10 iterations
for j in range(0, 100):
    data = np.random.rand(16, 100).astype(np.float32)
    label = (np.random.rand(16) * 10).astype(np.int32)
    workspace.FeedBlob("data", data)
    workspace.FeedBlob("label", label)
    workspace.RunNet(m.name, 10)   # run for 10 times

```

结语：  
转载请注明出处：[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

# Caffe2 模型与数据集（Models and Datasets）[3]

[Models and Datasets](https://link.jianshu.com?t=https://caffe2.ai/docs/tutorial-models-and-datasets.html)  
这一节没什么有用的信息为了保证教程完整性，这里仍然保留这一节。  
这一节唯一提到的一点就是：  
Caffe2的模型文件后缀是：.pb2

结语：  
转载请注明出处：[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

# Caffe2 的基本数据结构（Basics of Caffe2 - Workspaces, Operators, and Nets）[4]

这篇文章主要介绍Caffe2的基本数据结构：

+   Workspaces
+   Operators
+   Nets

在开始之前最好先阅读以下[`Intro Turorial`](https://www.jianshu.com/p/f1092b1e6822)  
**首先**，导入`caffe2`。其中`core`和`worksapce`模块，这是必须的两个模块。如果你要使用Caffe2生成的`protocol buffers`，那么你也需要从`caffe2_pb2`中导入`caffe2_pb2`模块。

```
# We'll also import a few standard python libraries
from matplotlib import pyplot
import numpy as np
import time

# These are the droids you are looking for.
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

```

如果你看到一些警告：`Caffe2`不支持GPU。这说明，你正在跑的Caffe2仅仅编译了CPU模式。不用担心，Caffe2在CPU上也是可以运行的。

### Workspaces

让我们先来介绍`Workspace`，它包含了所有数据。如果你熟悉`Matlab` ，`worksapce`包含了所有你创建的`blob`并保存在内存中。现在，让我们考虑一个N维的blob，blob和numpy的矩阵很像，但是它是连续的。接下来，我们将展示blob实际上是一种能指向任何C++类型对象的指针。下面，我们来看看接口是什么样的的。

`Blobs()`函数可以打印workspace里面所有的blobs。`HasBlob`则用于查询worksapce里面是否存在某个blob。不过，目前为止，我们的workspace里面没有任何东西。

```
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
print("Workspace has blob 'X'? {}".format(workspace.HasBlob("X")))

```

`FeedBlob()`函数用于向worksapce里面传递blob。

```
X = np.random.randn(2, 3).astype(np.float32)
print("Generated X from numpy:\n{}".format(X))
workspace.FeedBlob("X", X)

```

打印出来的X如下：

```
Generated X from numpy:
[[-0.56927377 -1.28052795 -0.95808828]
 [-0.44225693 -0.0620895  -0.50509363]]

```

让我们看一下workspace里面的blob是什么样的。

```
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
print("Workspace has blob 'X'? {}".format(workspace.HasBlob("X")))
print("Fetched X:\n{}".format(workspace.FetchBlob("X")))

```

输出如下：

```
Current blobs in the workspace: [u'X']
Workspace has blob 'X'? True
Fetched X:
[[-0.56927377 -1.28052795 -0.95808828]
 [-0.44225693 -0.0620895  -0.50509363]]

```

接着验证两个矩阵是否相等：

```
np.testing.assert_array_equal(X, workspace.FetchBlob("X"))

```

**注意**，如果你访问一个不存在的blob，将会引发一个错误：

```
try:
    workspace.FetchBlob("invincible_pink_unicorn")
except RuntimeError as err:
    print(err)

```

错误输出如下：

```
[enforce fail at pybind_state.cc:441] gWorkspace->HasBlob(name).

```

另外，有一个你目前可能还用不上的东西：你可以定义两个不同名字的workspace，并且在他们之间切换。不同workspace的bolb是相互分离的。你可以通过`CurrentWorkspace()`函数来访问当前的workspace。下面演示了如何切换不同的workspace和创建新的workspace。

```
print("Current workspace: {}".format(workspace.CurrentWorkspace()))
print("Current blobs in the workspace: {}".format(workspace.Blobs()))

# 切换到`gutentag` workspace，第二个参数`True`表示，如果`gutentag`不存在，则创建一个。
workspace.SwitchWorkspace("gutentag", True)

# 现在重新打印，注意到当前的workspace是`gutentag`，并且其中不包含任何东西。
print("Current workspace: {}".format(workspace.CurrentWorkspace()))
print("Current blobs in the workspace: {}".format(workspace.Blobs()))

```

程序输出：

```
Current workspace: default
Current blobs in the workspace: ['X']
Current workspace: gutentag
Current blobs in the workspace: []

```

重新切换回到`default`workspace

```
workspace.SwitchWorkspace("default")
print("Current workspace: {}".format(workspace.CurrentWorkspace()))
print("Current blobs in the workspace: {}".format(workspace.Blobs()))

```

并有如下输出：

```
Current workspace: default
Current blobs in the workspace: ['X']

```

最后，调用`ResetWorkspace()`函数可以清空**当前**的workspace的所有东西

```
workspace.ResetWorkspace()

```

### Operators

Caffe2中，operator就像函数一样。从C++的角度理解，operator全部从一个通用的接口继承而来，它们通过类型进行注册，所以，我们可以在运行时调用不同的操作。operator的接口定义在`caffe2/proto/caffe2.proto`文件中。Operator根据输出产生相应的输出。  
**记住**，在Caffe2的Python接口中，当我们说“创建一个operator”时，程序并没有跑起来，它只是创建了关于这个operator的protocol buffere，也就是定义了这个operator，但还没执行。之后，这个operator才会传递给C++接口禁止执行。如果你不明白什么是protobuf，那么你可以看下这个[链接](https://link.jianshu.com?t=https://developers.google.com/protocol-buffers/).  
\*\*1\. \*\*  
下面看一个实际例子：

```
# Create an operator.
op = core.CreateOperator(
    "Relu", # The type of operator that we want to run
    ["X"], # 输入 blobs 的名字的列表
    ["Y"], # A list of 输出 blobs by their names
)
# and we are done!

```

我们之前说到，创建op（operator）,事实上只是创建了一个protobuf对象。我们可以查看它的内容。

```
print("Type of the created op is: {}".format(type(op)))
print("Content:\n")
print(str(op))

```

输出如下：

```
Type of the created op is: <class 'caffe2.proto.caffe2_pb2.OperatorDef'>
Content:
input: "X"
output: "Y"
name: ""
type: "Relu"

```

现在跑起这个operator，我们首先需要向workspace中传入数据X，然后简单的调用`workspace.RunOperatorOnce(operator)`函数就可以。

```
workspace.FeedBlob("X", np.random.randn(2, 3).astype(np.float32))
workspace.RunOperatorOnce(op)

```

执行完后，让我们检查下这个operator是否正确操作。在这个操作中我们使用的是`Relu`函数。`Relu`函数在输入小于0时，取0，在输入大于0时，保持不变。

```
print("Current blobs in the workspace: {}\n".format(workspace.Blobs()))
print("X:\n{}\n".format(workspace.FetchBlob("X")))
print("Y:\n{}\n".format(workspace.FetchBlob("Y")))
print("Expected:\n{}\n".format(np.maximum(workspace.FetchBlob("X"), 0)))

```

输出如下,可以看到输出Y和你期望值一样，这个operator正确跑起来了：

```
Current blobs in the workspace: ['X', 'Y']
X:
[[ 1.03125858  1.0038228   0.0066975 ]
 [ 1.33142471  1.80271244 -0.54222912]]
Y:
[[ 1.03125858  1.0038228   0.0066975 ]
 [ 1.33142471  1.80271244  0.        ]]

Expected:
[[ 1.03125858  1.0038228   0.0066975 ]
 [ 1.33142471  1.80271244  0.        ]]

```

**2.**  
当然`Operator`也支持选项参数。选项参数通过key-value对确定。下面是一个简单的例子：创建一个tensor并且用高斯随机值填充它。

```
op = core.CreateOperator(
    "GaussianFill",
    [], # GaussianFill does not need any parameters.
    ["Z"],
    shape=[100, 100], # shape argument as a list of ints.
    mean=1.0,  # mean as a single float
    std=1.0, # std as a single float
)
print("Content of op:\n")
print(str(op))

```

看看输出：

```
Content of op:
output: "Z"
name: ""
type: "GaussianFill"
arg {
  name: "std"
  f: 1.0
}
arg {
  name: "shape"
  ints: 100
  ints: 100
}
arg {
  name: "mean"
  f: 1.0
}

```

然后我们跑起这个op,看看事情是否如期。

```
workspace.RunOperatorOnce(op)
temp = workspace.FetchBlob("Z")
pyplot.hist(temp.flatten(), bins=50)
pyplot.title("Distribution of Z")

```

image.png

  

没错，就是这样。

### Nets

`Net`其实是多个`operator`的集合，就像写程序时一行一行的命令。

让我们创建一个等价于下面Python代码的网络。

```
X = np.random.randn(2, 3)
W = np.random.randn(5, 3)
b = np.ones(5)
Y = X * W^T + b

```

Caffe2中的`core.net`是对`NetDef protocol buffer`的一个封装类。当创建一个网络时，这个对象完全是空的，除了拥有它的名字信息外。

```
net = core.Net("my_first_net")
print("Current network proto:\n\n{}".format(net.Proto()))

```

```
Current network proto:
name: "my_first_net"

```

接着创建一个blob，命名为“X”,使用高斯函数进行填充。

```
X = net.GaussianFill([], ["X"], mean=0.0, std=1.0, shape=[2, 3], run_once=0)
print("New network proto:\n\n{}".format(net.Proto()))

```

这时网络的结构如下

```
New network proto:
name: "my_first_net"
op {
  output: "X"
  name: ""
  type: "GaussianFill"
  arg {
    name: "std"
    f: 1.0
  }
  arg {
    name: "run_once"
    i: 0
  }
  arg {
    name: "shape"
    ints: 2
    ints: 3
  }
  arg {
    name: "mean"
    f: 0.0
  }
}

```

聪明的读者肯定想起了我们之前提到的`core.CreateOperator()`。事实上，当我们有了一个net，我们可以直接创建一个operator然后通过Python接口加到net中去。比如，你调用了`net.SomeOp`，这里的`SomeOp`是一个注册了的operator的字符串，因此上面的操作和下面等效。

```
op = core.CreateOperator("SomeOp", ...)
net.Proto().op.append(op)

```

> **译者注**:  
> 比如在我用`op = core.CreateOperator("GaussianFill",[], ["Z"],shape=[100, 100],mean=1.0, std=1.0)`创建了一个op，op的type为“GaussianFill”，这是一个注册了的类型。然后再调用`net.Proto().op.append(op)`把这个op添加到网络中去。  
> 以上的操作可以同过net来调用直接实现。直接使用op的type string---“GaussianFill”作为函数名字，net.GaussianFill(\[\], \["X"\], mean=0.0, std=1.0, shape=\[2, 3\], run_once=0)。

当然，读者可能感到困惑，X是什么？X是一个 `BlobReference`，这个引用包含两样东西：  

+  名字，可以通过`str(X)`来访问得到  

+  它是哪个net创建的，记录在其中的变量`_from_net`  
现在让我们验证它。同样记住，我们还没有跑任何东西，所以X只是个符号，里面什么也没有。别只望它会输出什么值。

```
print("Type of X is: {}".format(type(X)))
print("The blob name is: {}".format(str(X)))

```

```
Type of X is: <class 'caffe2.python.core.BlobReference'>
The blob name is: X

```

让我们继续创建W和b.

```
W = net.GaussianFill([], ["W"], mean=0.0, std=1.0, shape=[5, 3], run_once=0)
b = net.ConstantFill([], ["b"], shape=[5,], value=1.0, run_once=0)

```

现在一个简单的代码：**Note**由于`BlonReference`对象知道它由什么网络创建的，所以除了从net中创建op，你还可以通过`BlobReference`创建op。因此，我们可以通过如下方式创建FC操作。

```
Y = X.FC([W, b], ["Y"])

```

事实上，在底下，`X.FC(...)`只是简单的委托`net.FC`来实现，`X.FC()`会将X作为op的第一个输入。所以上面的操作其实等价于下面的：

```
Y = net.FC([X, W, b], ["Y"])

```

现在让我们看下当前这个网络。

```
print("Current network proto:\n\n{}".format(net.Proto()))

```

```
Current network proto:
name: "my_first_net"
op {
  output: "X"
  name: ""
  type: "GaussianFill"
  arg {
    name: "std"
    f: 1.0
  }
  arg {
    name: "run_once"
    i: 0
  }
  arg {
    name: "shape"
    ints: 2
    ints: 3
  }
  arg {
    name: "mean"
    f: 0.0
  }
}
op {
  output: "W"
  name: ""
  type: "GaussianFill"
  arg {
    name: "std"
    f: 1.0
  }
  arg {
    name: "run_once"
    i: 0
  }
  arg {
    name: "shape"
    ints: 5
    ints: 3
  }
  arg {
    name: "mean"
    f: 0.0
  }
}
op {
  output: "b"
  name: ""
  type: "ConstantFill"
  arg {
    name: "run_once"
    i: 0
  }
  arg {
    name: "shape"
    ints: 5
  }
  arg {
    name: "value"
    f: 1.0
  }
}
op {
  input: "X"
  input: "W"
  input: "b"
  output: "Y"
  name: ""
  type: "FC"
}

```

是不是觉得太过冗长？GOOD~让我们尝试下把它变成一个图。用ipython显示。

```
from caffe2.python import net_drawer
from IPython import display
graph = net_drawer.GetPydotGraph(net, rankdir="LR")
display.Image(graph.create_png(), width=800)

```

image.png

  
目前为止，我们已经定义了一个`Net`，但是并没有执行任何东西。记住，上面的net只是一个protobuf，仅仅定义了网路的结构。当我们真正跑起这个网络时，底层发生的事件如下。  

+  实例化protobuf中定义的C++net 对象  

+  调用实例化后的net的Run()函数  
在我们进行任何操作前，我们应该先使用`ResetWorkspace()`清空workspace里的东  
西。  
**NOTE**有两种方式通过python来跑一个网络。我们选择第一种来展示。

1.  使用 `workspace.RunNetOnce()`
2.  第二种更复杂点：需要两步，a) 调用`workspace.CreateNet()`创建C++net对象，b)使用`workspace.RunNet()`,这步需要传递网络的名字作为参数。

**第一种**

```
workspace.ResetWorkspace()
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
workspace.RunNetOnce(net)
print("Blobs in the workspace after execution: {}".format(workspace.Blobs()))
# Let's dump the contents of the blobs
for name in workspace.Blobs():
    print("{}:\n{}".format(name, workspace.FetchBlob(name)))

```

输出如下：

```
Current blobs in the workspace: []
Blobs in the workspace after execution: ['W', 'X', 'Y', 'b']
W:
[[-0.96537346  0.42591459  0.66788739]
 [-0.47695673  2.25724339 -0.10370601]
 [-0.20327474 -3.07469416  0.47715324]
 [-1.62159526  0.73711687 -1.42365313]
 [ 0.60718107 -0.50448036 -1.17132831]]
X:
[[-0.99601173 -0.61438894  0.10042733]
 [ 0.23359862  0.15135486  0.77555442]]
Y:
[[ 1.76692021  0.07781416  3.13944149  2.01927781  0.58755434]
 [ 1.35693741  1.14979863  0.85720366 -0.37135673  0.15705228]]
b:
[ 1.  1.  1.  1.  1.]

```

**第二种**  
现在尝试第二种方法去创建这个网络，并跑起它。

```
workspace.ResetWorkspace()
print("Current blobs in the workspace: {}".format(workspace.Blobs()))
workspace.CreateNet(net)
workspace.RunNet(net.Proto().name)#传入名字
print("Blobs in the workspace after execution: {}".format(workspace.Blobs()))
for name in workspace.Blobs():
    print("{}:\n{}".format(name, workspace.FetchBlob(name)))

```

输出

```
Current blobs in the workspace: []
Blobs in the workspace after execution: ['W', 'X', 'Y', 'b']
W:
[[-0.29295802  0.02897477 -1.25667715]
 [-1.82299471  0.92877913  0.33613944]
 [-0.64382178 -0.68545657 -0.44015241]
 [ 1.10232282  1.38060772 -2.29121733]
 [-0.55766547  1.97437167  0.39324901]]
X:
[[-0.47522315 -0.40166432  0.7179445 ]
 [-0.8363331  -0.82451206  1.54286408]]
Y:
[[ 0.22535783  1.73460138  1.2652775  -1.72335696  0.7543118 ]
 [-0.71776152  2.27745867  1.42452145 -4.59527397  0.4452306 ]]
b:
[ 1.  1.  1.  1.  1.]

```

`RunNetOnce()`和`RunNet()`之间有不少差异，其中最大的差异就是计算耗时。因为`RunNetOnce()`涉及到protobuf的序列化，和实例化网络。这可能会使用很长时间。让我们来看下开销。

```
# It seems that %timeit magic does not work well with
# C++ extensions so we'll basically do for loops
start = time.time()
for i in range(1000):
    workspace.RunNetOnce(net)
end = time.time()
print('Run time per RunNetOnce: {}'.format((end - start) / 1000))

start = time.time()
for i in range(1000):
    workspace.RunNet(net.Proto().name)
end = time.time()
print('Run time per RunNet: {}'.format((end - start) / 1000))

```

输出如下：

```
Run time per RunNetOnce: 0.000364284992218
Run time per RunNet: 4.42600250244e-06

```

可以看到RunNet()更快。

结语：以上就是Caffe2的Python接口的一些主要部件。装载请注明出处：  
[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

# Caffe2 玩玩回归（Toy Regression）[5]

### 前言

这一节将讲述如何使用Caffe2的特征进行简单的线性回归学习。主要分为以下几步：  

+  生成随机数据作为模型的输入  

+  用这些数据创建网络  

+  自动训练模型  

+  查看梯度递减的结果和学习过程中网络参数的变化  
ipython notebook教程请看[这里](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Toy_Regression.ipynb)  
**译者注**：如果图片看不清，可以保存到本地查看。

这是一个快速的例子，展示如何使用前面的基础教程进行快速的尝试用CNN进行回归。我们要解决的问题非常简单，输入是二维的x,输出是一维的y，权重w=\[2.0,1.5\]，偏置b=0.5。所以生成ground truth的等式是`y=wx+b`。

在这个教程中，我们将会使用Caffe2的op生成训练数据。注意，这和你日常训练工作不同：在真实的训练中，训练数据一般从外部源载入，比如Caffe的DB数据库，或者Hive表。我们将会在MNIST的例程中讲到。

这个例程中，每一个Caffe2 的op将会写得非常详细，所以会显得太多繁杂。但是在MNIST例程中，我们将使用CNN模型助手来构建CNN模型。

```
from caffe2.python import core, cnn, net_drawer, workspace, visualize
import numpy as np
from IPython import display
from matplotlib import pyplot

```

### 声明计算图

这里，我们声明两个图：一个用于初始化计算中将会用到的变量参数和常量，另外一个作为主图将会用于跑起梯度下降，也就是训练。（**译者注**：不明白为啥叫做计算图（computation graphs），其实看代码和前一个教程的一样，就是创建两个net，一个用于初始化参数，一个用于训练。）

**首先**，初始化网络：网络的名字不重要。我们基本上把初始化代码放在一个net中，这样，我们就可以调用`RunNetOnce()`函数来执行。我们分离`init_net`的原因是，这些操作在整个训练的过程中只需要执行一次。

```
init_net = core.Net("init")
# ground truth 参数.
W_gt = init_net.GivenTensorFill( [], "W_gt", shape=[1, 2], values=[2.0, 1.5])
B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
# Constant value ONE is used in weighted sum when updating parameters.
ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
# ITER是迭代的次数.
ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0, dtype=core.DataType.INT32)

# 随机初始化权重，范围在[-1,1]，初始化偏置为0
W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
print('Created init net.')

```

上面代码创建并初始化了`init_net`网络。主训练网络如下，我们展示了创建的的每一步。  

+  前向传播产生loss  

+  通过自动微分进行后向传播  

+  使用标准的SGD进行参数更新

```
train_net = core.Net("train")
# First, 生成随机的样本X和创建ground truth.
X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0, run_once=0)
Y_gt = X.FC([W_gt, B_gt], "Y_gt")
# 往ground truth添加高斯噪声
noise = train_net.GaussianFill([], "noise", shape=[64, 1], mean=0.0, std=1.0, run_once=0)
Y_noise = Y_gt.Add(noise, "Y_noise")
#注意到不需要讲梯度信息传播到 Y_noise层,
#所以使用StopGradient 函数告诉偏微分算法不需要做这一步
Y_noise = Y_noise.StopGradient([], "Y_noise")

# 线性回归预测
Y_pred = X.FC([W, B], "Y_pred")

# 使用欧拉损失并对batch进行平均
dist = train_net.SquaredL2Distance([Y_noise, Y_pred], "dist")
loss = dist.AveragedLoss([], ["loss"])

```

现在让我们看看网络是什么样子的。从下面的图可以看到，主要包含四部分。  

+  随机生成X  

+  使用`W_gt`,`B_gt`和`FC`操作生成grond truth `Y_gt`  

+  使用当前的参数W和B进行预测  

+  比较输出和计算损失

```
graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
display.Image(graph.create_png(), width=800)

```

  

现在，和其他框架相似，Caffe2允许我们自动地生成梯度操作，让我们试一下，并看看计算图有什么变化。

```
# Get gradients for all the computations above.
gradient_map = train_net.AddGradientOperators([loss])
graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
display.Image(graph.create_png(), width=800)

```

  

一旦我们获得参数的梯度，我们就可以将进行SGD操作：获得当前step的学习率，更参数。在这个例子中，我们没有做任何复杂的操作，只是简单的SGD。

```
# 迭代数增加1.
train_net.Iter(ITER, ITER)
# 根据迭代数计算学习率.
LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=20, gamma=0.9)
# 权重求和
train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)

graph = net_drawer.GetPydotGraph(train_net.Proto().op, "train", rankdir="LR")
display.Image(graph.create_png(), width=800)

```

再次展示计算图

  

  

既然我们创建了网络，那么跑起来

```
workspace.RunNetOnce(init_net)
workspace.CreateNet(train_net)

```

在我们开始训练之前，先来看看参数：

```
print("Before training, W is: {}".format(workspace.FetchBlob("W")))
print("Before training, B is: {}".format(workspace.FetchBlob("B")))

```

参数初始化如下

```
Before training, W is: [[-0.77634162 -0.88467366]]
Before training, B is: [ 0.]

```

训练：

```
for i in range(100):
    workspace.RunNet(train_net.Proto().name)

```

迭代100次后，查看参数：

```
print("After training, W is: {}".format(workspace.FetchBlob("W")))
print("After training, B is: {}".format(workspace.FetchBlob("B")))

print("Ground truth W is: {}".format(workspace.FetchBlob("W_gt")))
print("Ground truth B is: {}".format(workspace.FetchBlob("B_gt")))

```

参数如下：

```
After training, W is: [[ 1.95769441  1.47348857]]
After training, B is: [ 0.45236012]
Ground truth W is: [[ 2.   1.5]]
Ground truth B is: [ 0.5]

```

看起来相当简单是不是？让我们再近距离看看训练过程中参数的更新过程。为此，我们重新初始化参数，看看每次迭代参数的变化。记住，我们可以在任何时候从workspace中取出我们的blobs。

```
workspace.RunNetOnce(init_net)
w_history = []
b_history = []
for i in range(50):
    workspace.RunNet(train_net.Proto().name)
    w_history.append(workspace.FetchBlob("W"))
    b_history.append(workspace.FetchBlob("B"))
w_history = np.vstack(w_history)
b_history = np.vstack(b_history)
pyplot.plot(w_history[:, 0], w_history[:, 1], 'r')
pyplot.axis('equal')
pyplot.xlabel('w_0')
pyplot.ylabel('w_1')
pyplot.grid(True)
pyplot.figure()
pyplot.plot(b_history)
pyplot.xlabel('iter')
pyplot.ylabel('b')
pyplot.grid(True)

```

  

你可以发现非常典型的批梯度下降表现：由于噪声的影响，训练过程中存在波动。在Ipython notebook中跑多几次这个案例，你将会看到不同的初始化和噪声的影响。  
当然，这只是一个玩玩的例子，在MNIST例程中，我们将会看到一个更加真实的CNN训练的例子。

**译者注**： 转载请注明出处：[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

# Caffe2 图像预处理（Image Pre-Processing）[6]

学习如何使得图像符合预训练模型的需求，或者用其他数据集的图像来测试自己的模型。  

+  调整大小  

+  缩放  

+  HWC和CHW，数据通道交换  

+  RGB和BGR，颜色通道的交换  

+  Caffe2的图像预处理  
Ipython Notebook的教程在[这里获取](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Image_Pre-Processing_Pipeline.ipynb)  
在这一节中，我们将会展示如何从本地文件或网络链接载入一个图像，并能用于其他的教程和例子。当然，我们将继续深入多种预处理，这些预处理都是使用Caffe2时非常有必要的的。

### Mac OSx Prerequisites

首先，确保你有Python的这些模块。

```
sudo pip install scikit-image scipy matplotlib

```

然后，我们开始载入这些模块

```
%matplotlib inline
import skimage
import skimage.io as io
import skimage.transform
import sys
import numpy as np
import math
from matplotlib import pyplot
import matplotlib.image as mpimg
print("Required modules imported.")

```

* * *

### Test an Image

在下面的代码块中，用`IMAGE_LOCATION`去载入你想要测试的图像。改变其内容，并重新看看整个教程，你会看到对于不同的图片格式会有不同的处理。如果你想尝试自己的图像，把它改为你的图像路径或者远程URL。当你使用远程URL时，必须确保这个URL指向一个普通的图像文件类型和后缀，一些长的表示符或者字符串可能会导致程序中断。

* * *

### Color Issues

记住，如果你载入的图像来自智能手机，那么你可能会遇到图像颜色格式问题。在下面我们将会展示在RGB和BGR对一张图像的影响。确保图像数据和你想象中的一致。

###### Caffe Uses BGR Order

Caffe使用了OpenCV，而OpenCV处理图像是Blue-Green-Red (BGR) 形式的。而不是通用的RGB形式，所以在Caffe2中，图像的格式也是BGR。从长远来看，这种做法在很多方面是有益的，当你使用不同的计算机和库。但是这也是困惑的起源。

```
# 你可以载入本地图片或者远程连接

# 第一种方案，使用本地图像
#IMAGE_LOCATION = 'images/cat.jpg'
# 第二种线路使用网络图像，图像是一朵花
IMAGE_LOCATION = "https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg"
#第三种线路使用网络图像，网络图像有很多人
#IMAGE_LOCATION = "https://upload.wikimedia.org/wikipedia/commons/1/18/NASA_Astronaut_Group_15.jpg"
# 第四种使用一个网络图像，是一个竖图
#IMAGE_LOCATION = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Ducreux1.jpg"

img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)

# 显示原始图像
pyplot.figure()
pyplot.subplot(1,2,1)
pyplot.imshow(img)
pyplot.axis('on')
pyplot.title('Original image = RGB')

#交换颜色通道并显示交换后的的BGR图像
imgBGR = img[:, :, (2, 1, 0)]
pyplot.subplot(1,2,2)
pyplot.imshow(imgBGR)
pyplot.axis('on')
pyplot.title('OpenCV, Caffe2 = BGR')

```

  

由上面的例子中，你可以看到，不同的顺序是相当重要的。接下来的代码块中，我们将会图像转换为BGR顺序，这样Caffe2才能正确处理它。  
不，稍等。关于颜色还有些有趣的东西。

###### Caffe Prefers CHW Order

什么是CHW？还有HWC。这两个都是来源于图像处理。  

+  H:Height  

+  W:Width  

+  C:Channel  
深入了解图像在内存分配中的顺序。你可能注意到，当我们第一次载入图像时，我们进行了一些有趣的转换。这些数据转换就像把一幅图像当做一个魔方来玩。我们看到的是魔方的顶层，操作下面的层，可以改变看到的东西。

在GPU下，Caffe2需要的图像数据是CHW，在CPU下，一般需要的顺序是HWC。基本上，你需要CHW的顺序，并确保转换为CHW这步包含在你的图像预处理。把RGB转换为BGR，然后把HWC转换为CHW。这里的C就转换后的BGR。你可能会问，为什么呢？原因在于，在GPU上使用cuDNN库能获得非常大的加速，而cuDNN只使用CHW。总的来说，这样做能更快。

有了上面两步，你可能会觉得够了吧？不，你还是太年轻了。我们还需要resize（调整大小），crop（剪切），可能还需要些旋转和镜像。

* * *

### Rotation and Mirroring

来自智能手机的相片普遍存在着旋转或者镜像，有时，我们可以通过照片中的EXIF信息进行修正。但是并不是都这么幸运。

###### Library for Handling Mobile Images

下面展示的是旋转图像和镜像图像

```
# 对于这样的图像如何知道它是竖屏模式？
ROTATED_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/8/87/Cell_Phone_Tower_in_Ladakh_India_with_Buddhist_Prayer_Flags.jpg"
imgRotated = skimage.img_as_float(skimage.io.imread(ROTATED_IMAGE)).astype(np.float32)
pyplot.figure()
pyplot.imshow(imgRotated)
pyplot.axis('on')
pyplot.title('Rotated image')

#这种图像是给司机用后视镜看的
MIRROR_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/2/27/Mirror_image_sign_to_be_read_by_drivers_who_are_backing_up_-b.JPG"
imgMirror = skimage.img_as_float(skimage.io.imread(MIRROR_IMAGE)).astype(np.float32)
pyplot.figure()
pyplot.imshow(imgMirror)
pyplot.axis('on')
pyplot.title('Mirror image')

```

旋转

  

镜像

  

然我们做一些变换。同时，这些技巧可能能够帮到你，例如，你无法获取图像的EXIF信息，那么你可以对图像进行旋转，翻转，从而产生很多副本，对于这些图像，用你的模型全部跑一遍。当检测的置信度足够高时，找到了你需要的方向。  
代码如下：

```
#下面代码实现图像的左右翻转
imgMirror = np.fliplr(imgMirror)
pyplot.figure()
pyplot.imshow(imgMirror)
pyplot.axis('off')
pyplot.title('Mirror image')

```

镜像

```
#逆时针旋转90度
imgRotated = np.rot90(imgRotated)
pyplot.figure()
pyplot.imshow(imgRotated)
pyplot.axis('off')
pyplot.title('Rotated image')

```

旋转

* * *

### Sizing

下面的例子先将图像resize到256x256大小，然后从中剪切出224x224大小，因为网络的输入大小是224x224。

```
#模型的输入是224x224大小，因此需要resize或者crop
# (1) Resize 图像 256*256, 然后剪切中心部分
input_height, input_width = 224, 224
print("Model's input shape is %dx%d") % (input_height, input_width)
img256 = skimage.transform.resize(img, (256, 256))
pyplot.figure()
pyplot.imshow(img256)
pyplot.axis('on')
pyplot.title('Resized image to 256x256')
print("New image shape:" + str(img256.shape))

```

输出

```
Model's input shape is 224x224
New image shape:(256, 256, 3)

```

resize

  

注意resize有可能在一定程度上扭曲图像。你在测试时必须考虑这个问题，因为这会影响到你的模型输出的结果。花和动物被拉长或者压缩一点可能不会太大问题。但是面部特征就不一定行了。现在尝试另一种缩放图像的策略，并保持图像的比例不变。

###### Rescaling

保持图像的比例关系，并将最小的一边缩放到和网络的输入大小一致。在我们的例子中，网络的输入是224x224。

+   横向（Landscape）：限制高度进行resize
+   纵向（Portrait）：限制宽度进行resize

```
print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
print("Model's input shape is %dx%d") % (input_height, input_width)
aspect = img.shape[1]/float(img.shape[0])#宽/高
print("Orginal aspect ratio: " + str(aspect))
if(aspect>1):
    # 横向 - 宽图像
    res = int(aspect * input_height)#译者认为这里应该为input_width
    imgScaled = skimage.transform.resize(img, (input_width, res))#译者认为这里应该为input_height
if(aspect<1):
    # 竖向 - 高图像
    res = int(input_width/aspect)#译者认为这里应该为input_height
    imgScaled = skimage.transform.resize(img, (res, input_height))#译者认为这里应该为input_width
if(aspect == 1):
    imgScaled = skimage.transform.resize(img, (input_width, input_height))#译者认为这里应该为 input_height, input_width
pyplot.figure()
pyplot.imshow(imgScaled)
pyplot.axis('on')
pyplot.title('Rescaled image')
print("New image shape:" + str(imgScaled.shape) + " in HWC")

```

输出

```
Original image shape:(751, 1280, 3) and remember it should be in H, W, C!
Model's input shape is 224x224
Orginal aspect ratio: 1.70439414115
New image shape:(224, 381, 3) in HWC

```

image.png

###### Cropping

这里有很多策略可以使用。我们可以比例不变的将图像缩小到一边大小符合网络的输入，然后从图像中间剪切出一块。但是如果不进行缩放，可能只能剪切到图像中花的一部分。所以我们还是需要缩放。  
下面我们提供三种剪切的策略：

1.  直接从图像中间取出你需要的大小的patch
2.  resize到一个很接近网络输入大小的正方形，然后从中间抓取
3.  保持图像比例不变的缩放，然后从中间截取一部分

```
# 傻瓜式的从中间剪切
print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)   #python中//表示取结果的整数
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

pyplot.figure()
# Original image
imgCenter = crop_center(img,224,224)
pyplot.subplot(1,3,1)
pyplot.imshow(imgCenter)
pyplot.axis('on')
pyplot.title('Original')

# 从256x256的变形图像中剪切中间的224x224
img256Center = crop_center(img256,224,224)
pyplot.subplot(1,3,2)
pyplot.imshow(img256Center)
pyplot.axis('on')
pyplot.title('Squeezed')

# Scaled image
imgScaledCenter = crop_center(imgScaled,224,224)
pyplot.subplot(1,3,3)
pyplot.imshow(imgScaledCenter)
pyplot.axis('on')
pyplot.title('Scaled')

```

```
Original image shape:(751, 1280, 3) and remember it should be in H, W, C!

```

注意：内存上保存始终是H,W,C  
图像输出：

  

三种剪切方法

  

看起来好像最后一个比较好。第二种方法也不差，不过，这和很难说，要在你的模型上进行大批量测试才知道。如果你的模型在训练时使用不同比例的图像，并且直接将他们压缩到一个正方形，那么久而久之，你的模型将从压缩图像上学到那些物体被压缩时的样子，所以也能做出判断。但是如果你的模型专注于细节，比如面部特征，特征点，或者一些非常细微的元素，那么图像信息的丢失和变形将会带来非常大的误差。

**更好的策略**  
更好的方法是，把你的图像缩放到最接近真实数据，然后在图像边缘填补信息，填补的信息不能对你的模型产生影响，也就是你的模型会忽略掉这些信息。这个方法，我们会在另外一个教程中给出，因为，这个教程已经讲了不少了。

###### Upscaling

如果你想要跑的图像很小，怎么办？在我们的例子中，我们网络的输入是224x224，但是如果遇到下面的128x128的图像大小呢？  

  
最常用的方法就是，用skimage的工具把一个小的正方形图像变到一个大的正方形图像。`resize`的默认参数是1，对应着使用双线性插值。

```
imgTiny = "images/Cellsx128.png"
imgTiny = skimage.img_as_float(skimage.io.imread(imgTiny)).astype(np.float32)
print "Original image shape: ", imgTiny.shape
imgTiny224 = skimage.transform.resize(imgTiny, (224, 224))
print "Upscaled image shape: ", imgTiny224.shape
# Plot original
pyplot.figure()
pyplot.subplot(1, 2, 1)
pyplot.imshow(imgTiny)
pyplot.axis('on')
pyplot.title('128x128')
# Plot upscaled
pyplot.subplot(1, 2, 2)
pyplot.imshow(imgTiny224)
pyplot.axis('on')
pyplot.title('224x224')

```

```
Original image shape:  (128, 128, 4)
Upscaled image shape:  (224, 224, 4)

```

  
看到没，输出是 (224, 224, 4)。等等，为什么是4？前面所有例子都是3。当我们使用一个`png`文件时，它是由四个通道的。第四个通道代表的是‘模糊度’或者‘透明度’.无论怎么样，我们仍然能很好地处理它，不过，要留意这个通道数。现在让我们先转换成CHW，然后放大图像。

```
imgTiny = "images/Cellsx128.png"
imgTiny = skimage.img_as_float(skimage.io.imread(imgTiny)).astype(np.float32)
print "Image shape before HWC --> CHW conversion: ", imgTiny.shape
#交换坐标系HWC to CHW
imgTiny = imgTiny.swapaxes(1, 2).swapaxes(0, 1)
print "Image shape after HWC --> CHW conversion: ", imgTiny.shape
imgTiny224 = skimage.transform.resize(imgTiny, (224, 224))
print "Image shape after resize: ", imgTiny224.shape
try:
    pyplot.figure()
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(imgTiny)#交换顺序后无法显示
    pyplot.axis('on')
    pyplot.title('128x128')
except:
    print "Here come bad things!"
    # 如果你想看到错误，反注释掉下面一行
    #raise

```

  

什么都没显示，对吧,因为存储顺序调换了。但是通道数仍然是4.

现在让我们展示一个例子，一个比你网络输入小的图像，并且不是正方形的。来自于一个只能给出矩形图像的显微镜。

```
imgTiny = "images/Cellsx128.png"
imgTiny = skimage.img_as_float(skimage.io.imread(imgTiny)).astype(np.float32)
imgTinySlice = crop_center(imgTiny, 128, 56)
# Plot original
pyplot.figure()
pyplot.subplot(2, 1, 1)
pyplot.imshow(imgTiny)
pyplot.axis('on')
pyplot.title('Original')
# Plot slice
pyplot.figure()
pyplot.subplot(2, 2, 1)
pyplot.imshow(imgTinySlice)
pyplot.axis('on')
pyplot.title('128x56')
# Upscale?
print "Slice image shape: ", imgTinySlice.shape
imgTiny224 = skimage.transform.resize(imgTinySlice, (224, 224))
print "Upscaled slice image shape: ", imgTiny224.shape
# Plot upscaled
pyplot.subplot(2, 2, 2)
pyplot.imshow(imgTiny224)
pyplot.axis('on')
pyplot.title('224x224')

```

```
Slice image shape:  (56, 128, 4)
Upscaled slice image shape:  (224, 224, 4)

```

通道数没变。

  

  

  

这是一个非常严重的错误，例如正常的细胞都是接近圆形，而病变细胞则是镰刀形的。在这种情况下，你怎么办？这很依赖于你的模型和你的模型是如何训练出来的。在某些情况下，可以通过给图像填充白色或者黑色的，或者噪声的边缘解决这个问题。

下面我们继续讨论，我们已经说过BGR和CHW的问题了，但是在caffe2中还需要考虑一个就是`batch term`，也就是N，图像的个数。

###### Final Preprocessing and the Batch Term

```
# 如果你想尝试不同策略的剪切
# swap out imgScaled with img (original) or img256 (squeezed)
imgCropped = crop_center(imgScaled,224,224)
print "Image shape before HWC --> CHW conversion: ", imgCropped.shape
# (1)HWC->CHW
imgCropped = imgCropped.swapaxes(1, 2).swapaxes(0, 1)
print "Image shape after HWC --> CHW conversion: ", imgCropped.shape

pyplot.figure()
for i in range(3):
    # pyplot  subplot 索引和MATLAB的一样，从1开始
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(imgCropped[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))

# (2) RGB->BGR
imgCropped = imgCropped[(2, 1, 0), :, :]
print "Image shape after BGR conversion: ", imgCropped.shape
# 以下代码后面用到，现在没用
# (3) 减均值，由于skimage 读取的图像在[0,1]之间，所以我们需要乘以255，使像素范围回到[0,255]
#mean_file = os.path.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
#mean = np.load(mean_file).mean(1).mean(1)
#img = img * 255 - mean[:, np.newaxis, np.newaxis]

pyplot.figure()
for i in range(3):
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(imgCropped[i])
    pyplot.axis('off')
    pyplot.title('BGR channel %d' % (i+1))
# (4)最后由于Caffe2要求输入要有一个batch ,所以我们可以一次传递多张图像，我们仅仅让batch size=1,还要保证数据类型是 np.float32
imgCropped = imgCropped[np.newaxis, :, :, :].astype(np.float32)
print 'Final input shape is:', imgCropped.shape

```

输出：

```
Image shape before HWC --> CHW conversion:  (224, 224, 3)
Image shape after HWC --> CHW conversion:  (3, 224, 224)
Image shape after BGR conversion:  (3, 224, 224)
Final input shape is: (1, 3, 224, 224)

```

RGB

  

BGR

  

在上面的输出中，你应该注意到如下变化：

1.  HWC->CHW,图像的通道数3，由最后移到了前面
2.  RGB->BGR,蓝色和红色分量进行了交换
3.  输入数据的最后形状，在前面添加了batch size。所以数据格式是(1, 3, 224, 224)
    +   1是图像的个数
    +   3是图像的通道数
    +   224是高
    +   224是宽

这一教程到此结束。转载请注明出处：[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

# Caffe2 载入预训练模型（Loading Pre-Trained Models）[7]

这一节我们主要讲述如何使用预训练模型。Ipython notebook链接在[这里](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Loading_Pretrained_Models.ipynb)。

### 模型下载

你可以去[Model Zoo](https://link.jianshu.com?t=https://caffe2.ai/docs/zoo)下载预训练好的模型，或者使用Caffe2的`models.download`模块获取预训练的模型。`caffe2.python.models.download`需要模型的名字所谓参数。你可以去看看有什么模型可用，然后替换下面代码中的`squeezenet`。

```
python -m caffe2.python.models.download -i squeezenet

```

**译者注**：如果不明白为什么用python -m 执行，可以看看这个[帖子](https://link.jianshu.com?t=http://www.tuicool.com/articles/jMzqYzF)。  
如果上面下载成功，那么你应该下载了 squeezenet到你的文件夹中。如果你使用i那么模型文件将下载到`/caffe2/python/models`文件夹中。当然，你也可以下载所有模型文件：`git clone https://github.com/caffe2/models`。

### Overview

在这个教程中，我们将会使用`squeezenet`模型进行图片的目标识别。如果，你读了前面的预处理章节，那么你会看到我们使用rescale和crop对图像进行处理。同时做了CHW和BGR的转换，最后的图像数据是NCHW。我们也统计了图像均值，而不是简单地将图像减去128.  
你会发现载入预处理模型是相当简单的，仅仅需要几行代码就可以了。

1.  读取protobuf文件

```
with open("init_net.pb") as f:
     init_net = f.read()
 with open("predict_net.pb") as f:
     predict_net = f.read()   

```

2.  使用`Predictor`函数从protobuf中载入blobs数据

```
p = workspace.Predictor(init_net, predict_net)

```

3.  跑网络并获取结果

```
results = p.run([img])

```

返回的结果是一个多维概率的矩阵，每一行是一个百分比，表示网络识别出图像属于某一个物体的概率。当你使用前面那张花图来测试时，网络的返回应该告诉你超过95的概率是雏菊。

### Configuration

网络设置如下：

```
# 你安装caffe2的路径
CAFFE2_ROOT = "~/caffe2"
# 假设是caffe2的子目录
CAFFE_MODELS = "~/caffe2/caffe2/python/models"
#如果你有mean file，把它放在模型文件那个目录里面
%matplotlib inline
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import os
from caffe2.python import core, workspace
import urllib2
print("Required modules imported.")

```

传递图像的路径，或者网络图像的URL。物体编码参照Alex Net,比如“985”代表是“雏菊”。其他编码参照[这里](https://link.jianshu.com?t=https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes)。

```
IMAGE_LOCATION =  "https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg"

# 参数格式:  folder,      INIT_NET,          predict_net,         mean      , input image size
MODEL = 'squeezenet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 227

# AlexNet的物体编码
codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"
print "Config set!"

```

**处理图像**

```
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %dx%d") % (input_height, input_width)
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    pyplot.figure()
    pyplot.imshow(imgScaled)
    pyplot.axis('on')
    pyplot.title('Rescaled image')
    print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled
print "Functions set."

# set paths and variables from model choice and prep image
CAFFE2_ROOT = os.path.expanduser(CAFFE2_ROOT)
CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)

# 均值最好从训练集中计算得到
MEAN_FILE = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[3])
if not os.path.exists(MEAN_FILE):
    mean = 128
else:
    mean = np.load(MEAN_FILE).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]
print "mean was set to: ", mean

# 输入大小
INPUT_IMAGE_SIZE = MODEL[4]

# 确保所有文件存在
if not os.path.exists(CAFFE2_ROOT):
    print("Houston, you may have a problem.")
INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])
print 'INIT_NET = ', INIT_NET
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[2])
print 'PREDICT_NET = ', PREDICT_NET
if not os.path.exists(INIT_NET):
    print(INIT_NET + " not found!")
else:
    print "Found ", INIT_NET, "...Now looking for", PREDICT_NET
    if not os.path.exists(PREDICT_NET):
        print "Caffe model file, " + PREDICT_NET + " was not found!"
    else:
        print "All needed files found! Loading the model in the next block."

#载入一张图像
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print "After crop: " , img.shape
pyplot.figure()
pyplot.imshow(img)
pyplot.axis('on')
pyplot.title('Cropped')

# 转换为CHW
img = img.swapaxes(1, 2).swapaxes(0, 1)
pyplot.figure()
for i in range(3):
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))

#转换为BGR
img = img[(2, 1, 0), :, :]

# 减均值
img = img * 255 - mean

# 增加batch size
img = img[np.newaxis, :, :, :].astype(np.float32)
print "NCHW: ", img.shape

```

状态输出：

```
Functions set.
mean was set to:  128
INIT_NET =  /home/aaron/models/squeezenet/init_net.pb
PREDICT_NET =  /home/aaron/models/squeezenet/predict_net.pb
Found  /home/aaron/models/squeezenet/init_net.pb ...Now looking for /home/aaron/models/squeezenet/predict_net.pb
All needed files found! Loading the model in the next block.
Original image shape:(751, 1280, 3) and remember it should be in H, W, C!
Model's input shape is 227x227
Orginal aspect ratio: 1.70439414115
New image shape:(227, 386, 3) in HWC
After crop:  (227, 227, 3)
NCHW:  (1, 3, 227, 227)

```

  

  

  

既然图像准备好了，那么放进CNN里面吧。打开protobuf，载入到workspace中，并跑起网络。

```
#初始化网络
with open(INIT_NET) as f:
    init_net = f.read()
with open(PREDICT_NET) as f:
    predict_net = f.read()
p = workspace.Predictor(init_net, predict_net)

# 进行预测
results = p.run([img])

# 把结果转换为np矩阵
results = np.asarray(results)
print "results shape: ", results.shape

```

```
results shape:  (1, 1, 1000, 1, 1)

```

看到1000没。如果我们batch很大，那么这个矩阵将会很大，但是中间的维度仍然是1000。它记录着模型预测的每一个类别的概率。现在，让我们继续下一步。

```
results = np.delete(results, 1)#这句话不是很明白
index = 0
highest = 0
arr = np.empty((0,2), dtype=object)#创建一个0x2的矩阵？
arr[:,0] = int(10)#这是什么个意思？
arr[:,1:] = float(10)
for i, r in enumerate(results):
    # imagenet的索引从1开始
    i=i+1
    arr = np.append(arr, np.array([[i,r]]), axis=0)
    if (r > highest):
        highest = r
        index = i
print index, " :: ", highest
# top 3 结果
# sorted(arr, key=lambda x: x[1], reverse=True)[:3]

# 获取 code list
response = urllib2.urlopen(codes)
for line in response:
    code, result = line.partition(":")[::2]
    if (code.strip() == str(index)):
        print result.strip()[1:-2]

```

最后输出：

```
985  ::  0.979059
daisy

```

**译者注**：上面最后一段处理结果的代码，译者也不是很明白，有木有明白的同学在下面回复下？  
转载请注明出处：[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

# Caffe2 手写字符识别（MNIST - Create a CNN from Scratch）[8]

本教程创建一个小的神经网络用于手写字符的识别。我们使用MNIST数据集进行训练和测试。这个数据集的训练集包含60000张来自500个人的手写字符的图像，测试集包含10000张独立于训练集的测试图像。你可以参看本教程的[Ipython notebook](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/MNIST.ipynb)。

本节中，我们使用CNN的模型助手来创建网络并初始化参数。首先import所需要的依赖库。

```
%matplotlib inline
from matplotlib import pyplot
import numpy as np
import os
import shutil
from caffe2.python import core, cnn, net_drawer, workspace, visualize
# 如果你想更加详细的了解初始化的过程，那么你可以把caffe2_log_level=0 改为-1
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
caffe2_root = "~/caffe2"
print("Necessities imported!")

```

##### 数据准备

我们会跟踪训练过程的数据，并保存到一个本地的文件夹。我们需要先设置一个数据文件和根文件夹。在数据文件夹里，放置用于训练和测试的MNIST数据集。如果没有数据集，那么你可以到这里下载[MNIST Dataset](https://link.jianshu.com?t=https://caffe2.ai/docs/Models_and_Datasets.ipynb)，然后解压数据集和标签。

```
./make_mnist_db --channel_first --db leveldb --image_file ~/Downloads/train-images-idx3-ubyte --label_file ~/Downloads/train-labels-idx1-ubyte --output_file ~/caffe2/caffe2/python/tutorials/tutorial_data/mnist/mnist-train-nchw-leveldb

./make_mnist_db --channel_first --db leveldb --image_file ~/Downloads/t10k-images-idx3-ubyte --label_file ~/Downloads/t10k-labels-idx1-ubyte --output_file ~/caffe2/caffe2/python/tutorials/tutorial_data/mnist/mnist-test-nchw-leveldb

```

这段代码实现和上面的一样的功能

```
# 这部分将你的图像转换成leveldb
current_folder = os.getcwd()
data_folder = os.path.join(current_folder, 'tutorial_data', 'mnist')
root_folder = os.path.join(current_folder, 'tutorial_files', 'tutorial_mnist')
image_file_train = os.path.join(data_folder, "train-images-idx3-ubyte")
label_file_train = os.path.join(data_folder, "train-labels-idx1-ubyte")
image_file_test = os.path.join(data_folder, "t10k-images-idx3-ubyte")
label_file_test = os.path.join(data_folder, "t10k-labels-idx1-ubyte")

def DownloadDataset(url, path):
    import requests, zipfile, StringIO
    print "Downloading... ", url, " to ", path
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(label_file_train):
    DownloadDataset("https://s3.amazonaws.com/caffe2/datasets/mnist/mnist.zip", data_folder)

def GenerateDB(image, label, name):
    name = os.path.join(data_folder, name)
    print 'DB name: ', name
    syscall = "/usr/local/binaries/make_mnist_db --channel_first --db leveldb --image_file " + image + " --label_file " + label + " --output_file " + name
    print "Creating database with: ", syscall
    os.system(syscall)

# 生成leveldb
GenerateDB(image_file_train, label_file_train, "mnist-train-nchw-leveldb")
GenerateDB(image_file_test, label_file_test, "mnist-test-nchw-leveldb")

if os.path.exists(root_folder):
    print("Looks like you ran this before, so we need to cleanup those old workspace files...")
    shutil.rmtree(root_folder)

os.makedirs(root_folder)
workspace.ResetWorkspace(root_folder)

print("training data folder:"+data_folder)
print("workspace root folder:"+root_folder)

```

##### 模型创建

`CNNModelHelper`封装了很多函数，它能将参数初始化和真实的计算分成两个网络中实现。底层实现是，CNNModelHelper有两个网络`param_init_net`和`net`，这两个网络分别记录着初始化网络和主网络。为了模块化，我们将模型分割成多个不同的部分。  

+  数据输入（AddInput 函数）  

+  主要的计算部分（AddLeNetModel 函数）  

+  训练部分-梯度操作，参数更新等等 （AddTrainingOperators函数）  

+  记录数据部分，比如需要展示训练过程的相关数据（AddBookkeepingOperators 函数）

1.  **AddInput**会从一个DB中载入数据。我们将MNIST保存为像素值，并且我们用浮点数进行计算，所以我们的数据也必须是Float类型。为了数值稳定性，我们将图像数据归一化到\[0,1\]而不是\[0,255\]。注意，我们做的事in-place操作，会覆盖原来的数据，因为我们不需要归一化前的数据。准备数据这个操作，在后向传播时，不需要进行梯度计算。所以我们使用`StopGradient`来告诉梯度生成器：“不用将梯度传递给我。”

```
def AddInput(model, batch_size, db, db_type):
    # 载入数据和标签
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # 转化为 float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    #归一化到 [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # 后向传播不需要梯度
    data = model.StopGradient(data, data)
    return data, label
print("Input function created.")

```

输出

```
Input function created.

```

2.  **AddLeNetModel**输出`softmax`.

```
def AddLeNetModel(model, data):
    conv1 = model.Conv(data, 'conv1', 1, 20, 5)
    pool1 = model.MaxPool(conv1, 'pool1', kernel=2, stride=2)
    conv2 = model.Conv(pool1, 'conv2', 20, 50, 5)
    pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=2)
    fc3 = model.FC(pool2, 'fc3', 50 * 4 * 4, 500)
    fc3 = model.Relu(fc3, fc3)
    pred = model.FC(fc3, 'pred', 500, 10)
    softmax = model.Softmax(pred, 'softmax')
    return softmax
print("Model function created.")

```

```
Model function created.

```

3.  **AddTrainingOperators**函数函数用于添加训练操作。  
    AddAccuracy函数输出模型的准确率，我们会在下一个函数使用它来跟踪准确率。

```
def AddAccuracy(model, softmax, label):
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy
print("Accuracy function created.")

```

```
Accuracy function created.

```

首先添加一个op：LabelCrossEntropy，用于计算输入和lebel的交叉熵。这个操作在得到softmax后和计算loss前。输入是\[softmax, label\],输出交叉熵用xent表示。

```
xent = model.LabelCrossEntropy([softmax, label], 'xent')

```

AveragedLoss将交叉熵作为输入，并计算出平均损失loss

```
loss = model.AveragedLoss(xent, "loss")

```

AddAccuracy为了记录训练过程，我们使用AddAccuracy 函数来计算。

```
AddAccuracy(model, softmax, label)

```

接下来这步至关重要：我们把所有梯度计算添加到模型上。梯度是根据我们前面的loss计算得到的。

```
model.AddGradientOperators([loss])

```

然后进入迭代

```
ITER = model.Iter("iter")

```

更新学习率使用策略是`lr = base_lr * (t ^ gamma)`,注意我们是在最小化，所以基础学率是负数，这样我们才能向山下走。

```
LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 ) 
#ONE是一个在梯度更新阶段用的常量。只需要创建一次，并放在param_init_net中。
ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

```

现在对于每一和参数，我们做梯度更新。注意我们如何获取每个参数的梯度——CNNModelHelper保持跟踪这些信息。更新的方式很简单，是简单的相加：`param = param + param_grad * LR`

```
for param in model.params:
    param_grad = model.param_to_grad[param]
    model.WeightedSum([param, ONE, param_grad, LR], param)   

```

我们需要每隔一段时间检查参数。这可以通过`Checkpoint` 操作。这个操作有一个参数`every`表示每多少次迭代进行一次这个操作，防止太频繁去检查。这里，我们每20次迭代进行一次检查。

```
model.Checkpoint([ITER] + model.params, [],
               db="mnist_lenet_checkpoint_%05d.leveldb",
               db_type="leveldb", every=20)

```

然后我们得到整个AddTrainingOperators函数如下：

```
def AddTrainingOperators(model, softmax, label):
    # 计算交叉熵
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # 计算loss
    loss = model.AveragedLoss(xent, "loss")
    #跟踪模型的准确率
    AddAccuracy(model, softmax, label)
    #添加梯度操作
    model.AddGradientOperators([loss])
    # 梯度下降
    ITER = model.Iter("iter")
    # 学习率
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999 )
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # 梯度更新
    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.WeightedSum([param, ONE, param_grad, LR], param)
    # 每迭代20次检查一次
    # you may need to delete tutorial_files/tutorial-mnist to re-run the tutorial
    model.Checkpoint([ITER] + model.params, [],
                   db="mnist_lenet_checkpoint_%05d.leveldb",
                   db_type="leveldb", every=20)
print("Training function created.")

```

```
Training function created.

```

4.  \*\*AddBookkeepingOperators \*\*添加一些记录操作，这些操作不会影响训练过程。他们只是收集数据和打印出来或者写到log里面去。

```
def AddBookkeepingOperators(model):
    # 输出 blob的内容. to_file=1 表示输出到文件，文件保存的路径是 root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes 给出一些参数比如均值，方差，最大值，最小值
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
print("Bookkeeping function created")

```

5.  **定义网络**  
    现在让我们将真正创建模型。前面写的函数将真正被执行。回忆我们四步。

```
-data input  
-main computation
-training
-bookkeeping

```

在我们读进数据前，我们需要定义我们训练模型。我们将使用到前面定义的所有东西。我们将在MNIST数据集上使用NCHW的储存顺序。

```
train_model = cnn.CNNModelHelper(order="NCHW", name="mnist_train")
data, label = AddInput(train_model, batch_size=64,
              db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'), db_type='leveldb')
softmax = AddLeNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)
AddBookkeepingOperators(train_model)
# Testing model. 我们设置batch=100，这样迭代100次就能覆盖10000张测试图像
# 对于测试模型，我们需要数据输入 ，LeNetModel，和准确率三部分
#注意到init_params 设置为False，是因为我们从训练网络获取参数。
test_model = cnn.CNNModelHelper(order="NCHW", name="mnist_test", init_params=False)
data, label = AddInput(test_model, batch_size=100,
    db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'), db_type='leveldb')
softmax = AddLeNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)

# Deployment model. 我们仅需要LeNetModel 部分
deploy_model = cnn.CNNModelHelper(order="NCHW", name="mnist_deploy", init_params=False)
AddLeNetModel(deploy_model, "data")
#你可能好奇deploy_model的param_init_net 发生了什么，在这节中，我们没有使用它，
#因为在deployment 阶段，我们不会随机初始化参数，而是从本地载入。
print('Created training and deploy models.')


```

现在让我们用caffe2的可视化工具看看Training和Deploy模型是什么样子的。如果下面的命令运行失败，那可能是因为你的机器没有安装`graphviz`。可以用如下命令安装：

```
sudo yum install graphviz #ubuntu 用户sudo apt-get install graphviz 

```

图看起来可能很小，右键点击在新的窗口打开就能看清。

```
from IPython import display
graph = net_drawer.GetPydotGraph(train_model.net.Proto().op, "mnist", rankdir="LR")
display.Image(graph.create_png(), width=800)

```

  

现在上图展示了训练阶段的一切东西。白色的节点是blobs，绿色的矩形节点是operators.你可能留意到大规模的像火车轨道一样的平行线。这些依赖关系从前前向传播的blobs指向到后向传播的操作。  
让我们仅仅展示必要的依赖关系和操作。如果你细心看，你会发现，左半图式前向传播，右半图式后向传播，在最右边是一系列参数更新操作和summarization .

```
graph = net_drawer.GetPydotGraphMinimal(
    train_model.net.Proto().op, "mnist", rankdir="LR", minimal_dependency=True)
display.Image(graph.create_png(), width=800)

```

  

现在我们可以通过Python来跑起网络，记住，当我们跑起网络时，我们随时可以从网络中拿出blob数据，下面先来展示下如何进行这个操作。

我们重申一下，CNNModelHelper 类目前没有执行任何东西。他目前做的仅仅是声明网络，只是简单的创建了protocol buffers.例如我们可以展示网络一部分序列化的protobuf。

```
print(str(train_model.param_init_net.Proto())[:400] + '\n...')

```

当然，我们也可以把protobuf写到本地磁盘中去，这样可以方便的查看。你会发现这些protobuf和以前的Caffe网络定义很相似。

```
with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    fid.write(str(deploy_model.net.Proto()))
print("Protocol buffers files have been created in your root folder: "+root_folder)

```

现在，让我们进入训练过程。我们使用Python来训练。当然也可以使用C++接口来训练。这留在另一个教程讨论。

##### 训练网络

首先，初始化网络是必须的

```
workspace.RunNetOnce(train_model.param_init_net)

```

接着我们创建训练网络并，加载到workspace中去。

```
workspace.CreateNet(train_model.net)

```

然后设置迭代200次，并把准确率和loss保存到两个np矩阵中去

```
total_iters = 200
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)

```

网络和跟踪准确loss都配置好后，我们循环调用`workspace.RunNet`200次，需要传入的参数是`train_model.net.Proto().name`.每一次迭代，我们计算准确率和loss。

```
for i in range(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')

```

最后我们可以用`pyplot`画出结果。

```
# 参数初始化只需跑一次
workspace.RunNetOnce(train_model.param_init_net)
# 创建网络
workspace.CreateNet(train_model.net)
#设置迭代数和跟踪accuracy & loss
total_iters = 200
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# 我们迭代200次
for i in range(total_iters):
    workspace.RunNet(train_model.net.Proto().name)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
# 迭代完画出结果
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')

```

  

现我们可以进行抽取数据和预测了

```
#数据可视化
pyplot.figure()
data = workspace.FetchBlob('data')
_ = visualize.NCHW.ShowMultiple(data)
pyplot.figure()
softmax = workspace.FetchBlob('softmax')
_ = pyplot.plot(softmax[0], 'ro')
pyplot.title('Prediction for the first image')

```

  

  

还记得我们创建的test net吗？我们将跑一遍test net测试准确率。**注意，虽然test\_model的参数来自train\_model,但是仍然需要初始化test\_model.param\_init_net **。这次，我们只需要追踪准确率，并且只迭代100次。

```
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)
test_accuracy = np.zeros(100)
for i in range(100):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
pyplot.plot(test_accuracy, 'r')
pyplot.title('Acuracy over test batches.')
print('test_accuracy: %f' % test_accuracy.mean())

```

  
**译者注**：这里译者不是很明白，test\_model是如何从train\_model获取参数的？有明白的小伙伴希望能在评论区分享一下。

MNIST教程就此结束。希望本教程能向你展示一些Caffe2的特征。

转载请注明出处：[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

# Caffe2 创建你的专属数据集（Create Your Own Dataset）[9]

这一节尝试把你的数据转换成caffe2能够使用的形式。这个教程使用Iris的数据集。你可以点击[这里](https://link.jianshu.com?t=https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/create_your_own_dataset.ipynb)查看Ipython Notebook教程。

#### DB数据格式

Caffe2使用二进制的DB格式来保存数据。Caffe2 DB其实是键-值存储方式的一个美名而已。在键-值（key-value）存储方式里，键是随机生成的，所以batches是独立同分布的。而值（Value）则是真正的数据，他们包含着训练过程中真正用到的数据。所以，DB中保存的数据格式就像下面这样:

> key1 value1 key2 value2 key3 value3 ...

在DB中，他把keys和values看成strings。你可以用TensorProtos protobuf来将你要保存的东西保存成DB数据结构。一个TensorProtos protobuf封装了Tensor（多维矩阵），和它的数据类型，形状信息。然后，你可以通过TensorProtosDBInput操作来载入数据到SGD训练过程中。

#### 准备自己的数据

这里，我们向你展示如何创建自己的数据集。为此，我们将会使用UCI Iris数据集。这是一个非常受欢迎的经典的用于分类鸢尾花的数据集。它包含4个代表花的外形特征的实数。这个数据集包含3种鸢尾花。你可以从这里下载[数据集](https://link.jianshu.com?t=https://archive.ics.uci.edu/ml/datasets/Iris)。

```
%matplotlib inline
import urllib2 # 用于从网上下载数据集
import numpy as np
from matplotlib import pyplot
from StringIO import StringIO
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

```

```
WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
WARNING:root:Debug message: No module named caffe2_pybind11_state_gpu
#如果你在Mac OS下使用homebrew，你可能会遇到一个错误： malloc_zone_unregister() 函数失败.这不是Caffe2的问题，而是因为 homebrew leveldb 的内存分配不兼容. 但这不影响使用。

```

```
f = urllib2.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
raw_data = f.read()
print('Raw data looks like this:')
print(raw_data[:100] + '...')

```

输出：

```
Raw data looks like this:
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,...

```

```
#将特征保存到一个特征矩阵
features = np.loadtxt(StringIO(raw_data), dtype=np.float32, delimiter=',', usecols=(0, 1, 2, 3))
#把label存到一个特征矩阵中
label_converter = lambda s : {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}[s]
labels = np.loadtxt(StringIO(raw_data), dtype=np.int, delimiter=',', usecols=(4,), converters={4: label_converter})

```

在我们开始训练之前，最好将数据集分成训练集和测试集。在这个例子中，让我们随机打乱数据，用前100个数据做训练集，剩余50个数据做测试。当然你也可以用更加复杂的方式，例如使用交叉校验的方式将数据集分成多个训练集和测试集。关于交叉校验的更多信息，请看[这里](https://link.jianshu.com?t=http://scikit-learn.org/stable/modules/cross_validation.html)。

```
random_index = np.random.permutation(150)
features = features[random_index]
labels = labels[random_index]
train_features = features[:100]
train_labels = labels[:100]
test_features = features[100:]
test_labels = labels[100:]

```

```
legend = ['rx', 'b+', 'go']
pyplot.title("Training data distribution, feature 0 and 1")
for i in range(3):
    pyplot.plot(train_features[train_labels==i, 0], train_features[train_labels==i, 1], legend[i])
pyplot.figure()
pyplot.title("Testing data distribution, feature 0 and 1")
for i in range(3):
    pyplot.plot(test_features[test_labels==i, 0], test_features[test_labels==i, 1], legend[i])

```

  

现在，把数据放进Caffe2的DB中去。在这个DB中，我们将会使用`train_xxx`作为key，并对于每一个点使用一个TensorProtos对象去储存，一个TensorProtos包含两个tensor：一个是特征，一个是label。我们使用Caffe2的Python DB接口。

```
# 构建一个TensorProtos protobuf 
feature_and_label = caffe2_pb2.TensorProtos()
feature_and_label.protos.extend([
    utils.NumpyArrayToCaffe2Tensor(features[0]),
    utils.NumpyArrayToCaffe2Tensor(labels[0])])
print('This is what the tensor proto looks like for a feature and its label:')
print(str(feature_and_label))
print('This is the compact string that gets written into the db:')
print(feature_and_label.SerializeToString())

```

```
This is what the tensor proto looks like for a feature and its label:
protos {
  dims: 4
  data_type: FLOAT
  float_data: 5.40000009537
  float_data: 3.0
  float_data: 4.5
  float_data: 1.5
}
protos {
  data_type: INT32
  int32_data: 1
}
This is the compact string that gets written into the db:
��������̬@@@�@�?
���"��

```

现在真正写入DB中去

```
def write_db(db_type, db_name, features, labels):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    for i in range(features.shape[0]):
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(features[i]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])])
        transaction.put(
            'train_%03d'.format(i),
            feature_and_label.SerializeToString())
    # Close the transaction, and then close the db.
    del transaction
    del db

write_db("minidb", "iris_train.minidb", train_features, train_labels)
write_db("minidb", "iris_test.minidb", test_features, test_labels)

```

现在让我恩创建一个简单的网络，这个网络只包含一个简单的TensorProtosDBInput 操作，用来展示我们如何从创建好的DB中读入数据。

```
net_proto = core.Net("example_reader")
dbreader = net_proto.CreateDB([], "dbreader", db="iris_train.minidb", db_type="minidb")
net_proto.TensorProtosDBInput([dbreader], ["X", "Y"], batch_size=16)

print("The net looks like this:")
print(str(net_proto.Proto()))

```

```
The net looks like this:
name: "example_reader"
op {
  output: "dbreader"
  name: ""
  type: "CreateDB"
  arg {
    name: "db_type"
    s: "minidb"
  }
  arg {
    name: "db"
    s: "iris_train.minidb"
  }
}
op {
  input: "dbreader"
  output: "X"
  output: "Y"
  name: ""
  type: "TensorProtosDBInput"
  arg {
    name: "batch_size"
    i: 16
  }
}

```

创建网络

```
workspace.CreateNet(net_proto)

```

```
# 先跑一次，然后获取里面的数据
workspace.RunNet(net_proto.Proto().name)
print("The first batch of feature is:")
print(workspace.FetchBlob("X"))
print("The first batch of label is:")
print(workspace.FetchBlob("Y"))

# 再跑一次
workspace.RunNet(net_proto.Proto().name)
print("The second batch of feature is:")
print(workspace.FetchBlob("X"))
print("The second batch of label is:")
print(workspace.FetchBlob("Y"))

```

```
The first batch of feature is:
[[ 5.19999981  4.0999999   1.5         0.1       ]
 [ 5.0999999   3.79999995  1.5         0.30000001]
 [ 6.9000001   3.0999999   4.9000001   1.5       ]
 [ 7.69999981  2.79999995  6.69999981  2.        ]
 [ 6.5999999   2.9000001   4.5999999   1.29999995]
 [ 6.30000019  2.79999995  5.0999999   1.5       ]
 [ 7.30000019  2.9000001   6.30000019  1.79999995]
 [ 5.5999999   2.9000001   3.5999999   1.29999995]
 [ 6.5         3.          5.19999981  2.        ]
 [ 5.          3.4000001   1.5         0.2       ]
 [ 6.9000001   3.0999999   5.4000001   2.0999999 ]
 [ 6.          3.4000001   4.5         1.60000002]
 [ 5.4000001   3.4000001   1.70000005  0.2       ]
 [ 6.30000019  2.70000005  4.9000001   1.79999995]
 [ 5.19999981  2.70000005  3.9000001   1.39999998]
 [ 6.19999981  2.9000001   4.30000019  1.29999995]]
The first batch of label is:
[0 0 1 2 1 2 2 1 2 0 2 1 0 2 1 1]
The second batch of feature is:
[[ 5.69999981  2.79999995  4.0999999   1.29999995]
 [ 5.0999999   2.5         3.          1.10000002]
 [ 4.4000001   2.9000001   1.39999998  0.2       ]
 [ 7.          3.20000005  4.69999981  1.39999998]
 [ 5.69999981  2.9000001   4.19999981  1.29999995]
 [ 5.          3.5999999   1.39999998  0.2       ]
 [ 5.19999981  3.5         1.5         0.2       ]
 [ 6.69999981  3.          5.19999981  2.29999995]
 [ 6.19999981  3.4000001   5.4000001   2.29999995]
 [ 6.4000001   2.70000005  5.30000019  1.89999998]
 [ 6.5         3.20000005  5.0999999   2.        ]
 [ 6.0999999   3.          4.9000001   1.79999995]
 [ 5.4000001   3.4000001   1.5         0.40000001]
 [ 4.9000001   3.0999999   1.5         0.1       ]
 [ 5.5         3.5         1.29999995  0.2       ]
 [ 6.69999981  3.          5.          1.70000005]]
The second batch of label is:
[1 1 0 1 1 0 0 2 2 2 2 2 0 0 0 1]

```

至此，本节教程结束。  
转载请注明出处：[http://www.jianshu.com/c/cf07b31bb5f2](https://www.jianshu.com/c/cf07b31bb5f2)

