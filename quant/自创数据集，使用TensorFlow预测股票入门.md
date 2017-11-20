# 自创数据集，使用TensorFlow预测股票入门

> 来源：[机器之心](https://www.jiqizhixin.com/articles/2017-11-12)

> STATWORX 团队近日从 Google Finance API 中精选出了 S＆P 500 数据，该数据集包含 S＆P 500 的指数和股价信息。有了这些数据，他们就希望能利用深度学习模型和 500 支成分股价预测 S&P 500 指数。STATWORX 团队的数据集十分新颖，但只是利用四个隐藏层的全连接网络实现预测，读者也可以下载该数据尝试更加优秀的循环神经网络。

本文非常适合初学者了解如何使用 TensorFlow 构建基本的神经网络，它全面展示了构建一个 TensorFlow 模型所涉及的概念与模块。本文所使用的数据集可以直接下载，所以有一定基础的读者也可以尝试使用更强的循环神经网络处理这一类时序数据。

数据集地址：http://files.statworx.com/sp500.zip

**导入和预处理数据**

STATWORX 团队从服务器爬取股票数据，并将它们保存为 csv 格式的文件。该数据集包含 n=41266 分钟的记录，范围从 2017 年的 4 月到 8 月的 500 支股票和 S&P 500 指数，股票和股指的范围分布十分广。

```
# Import data
data = pd.read_csv('data_stocks.csv')
# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

```

该数据集已经经过了清理与预处理，即损失的股票和股指都通过 LOCF'ed 处理（下一个观测数据复制前面的），所以该数据集没有任何缺损值。

我们可以使用 pyplot.plot('SP500') 语句绘出 S&P 时序数据。

![](https://segmentfault.com/img/remote/1460000012074737)

_S&P 500 股指时序绘图_

**预备训练和测试数据**

该数据集需要被分割为训练和测试数据，训练数据包含总数据集 80% 的记录。该数据集并不需要扰乱而只需要序列地进行切片。训练数据可以从 2017 年 4 月选取到 2017 年 7 月底，而测试数据再选取剩下到 2017 年 8 月的数据。

```
# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

```

时序交叉验证有很多不同的方式，例如有或没有再拟合（refitting）而执行滚动式预测、或者如时序 bootstrap 重采样等更加详细的策略等。后者涉及时间序列周期性分解的重复样本，以便模拟与原时间序列相同周期性模式的样本，但这并不不是简单的复制他们的值。

**数据标准化**

大多数神经网络架构都需要标准化数据，因为 tanh 和 sigmoid 等大多数神经元的激活函数都定义在 [-1, 1] 或 [0, 1] 区间内。目前线性修正单元 ReLU 激活函数是最常用的，但它的值域有下界无上界。不过无论如何我们都应该重新缩放输入和目标值的范围，这对于我们使用梯度下降算法也很有帮助。缩放取值可以使用 sklearn 的 MinMaxScaler 轻松地实现。

```
# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_train)
scaler.transform(data_train)
scaler.transform(data_test)
# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]pycharm

```

注意，我们必须谨慎地确定什么时候该缩放哪一部分数据。比较常见的错误就是在拆分测试和训练数据集之前缩放整个数据集。因为我们在执行缩放时会涉及到计算统计数据，例如一个变量的最大和最小值。但在现实世界中我们并没有来自未来的观测信息，所以必须对训练数据按比例进行统计计算，并将统计结果应用于测试数据中。不然的话我们就使用了未来的时序预测信息，这常常令预测度量偏向于正向。

**TensorFlow 简介**

TensorFlow 是一个十分优秀的框架，目前是深度学习和神经网络方面用户最多的框架。它基于 C++的底层后端，但通常通过 Python 进行控制。TensorFlow 利用强大的静态图表征我们需要设计的算法与运算。这种方法允许用户指定运算为图中的结点，并以张量的形式传输数据而实现高效的算法设计。由于神经网络实际上是数据和数学运算的计算图，所以 TensorFlow 能很好地支持神经网络和深度学习。

总的来说，TensorFlow 是一种采用数据流图（data flow graphs），用于数值计算的开源软件库。其中 Tensor 代表传递的数据为张量（多维数组），Flow 代表使用计算图进行运算。数据流图用「结点」（nodes）和「边」（edges）组成的有向图来描述数学运算。「结点」一般用来表示施加的数学操作，但也可以表示数据输入的起点和输出的终点，或者是读取/写入持久变量（persistent variable）的终点。边表示结点之间的输入/输出关系。这些数据边可以传送维度可动态调整的多维数据数组，即张量（tensor）。

![](https://segmentfault.com/img/remote/1460000012074738)

_执行加法的简单计算图_

在上图中，两个零维张量（标量）将执行相加任务，这两个张量储存在两个变量 a 和 b 中。这两个值流过图形在到达正方形结点时被执行相加任务，相加的结果被储存在变量 c 中。实际上，a、b 和 c 可以被看作占位符，任何输入到 a 和 b 的值都将会相加到 c。这正是 TensorFlow 的基本原理，用户可以通过占位符和变量定义模型的抽象表示，然后再用实际的数据填充占位符以产生实际的运算，下面的代码实现了上图简单的计算图：

```
# Import TensorFlow
import tensorflow as tf
# Define a and b as placeholders
a = tf.placeholder(dtype=tf.int8)
b = tf.placeholder(dtype=tf.int8)
# Define the addition
c = tf.add(a, b)
# Initialize the graph
graph = tf.Session()
# Run the graph
graph.run(c, feed_dict{a: 5, b: 4})

```

如上在导入 TensorFlow 库后，使用 tf.placeholder() 定义两个占位符来预储存张量 a 和 b。随后定义运算后就能执行运算图得出结果。

**占位符**

正如前面所提到的，神经网络的初始源自占位符。所以现在我们先要定义两个占位符以拟合模型，X 包含神经网络的输入（所有 S&P 500 在时间 T=t 的股票价格），Y 包含神经网络的输出（S&P 500 在时间 T=t+1 的指数值）。

因此输入数据占位符的维度可定义为 [None, n_stocks]，输出占位符的维度为 [None]，它们分别代表二维张量和一维张量。理解输入和输出张量的维度对于构建整个神经网络十分重要。

```
# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

```

以上代码中的 None 指代我们暂时不知道每个批量传递到神经网络的数量，所以使用 None 可以保持灵活性。我们后面会定义控制每次训练时使用的批量大小 batch_size。

变量

除了占位符，变量是 TensorFlow 表征数据和运算的另一个重要元素。虽然占位符在计算图内通常用于储存输入和输出数据，但变量在计算图内部是非常灵活的容器，它可以在执行中进行修改与传递。神经网络的权重和偏置项一般都使用变量定义，以便在训练中可以方便地进行调整，变量需要进行初始化，后文将详细解释这一点。

该模型由四个隐藏层组成，第一层包含 1024 个神经元，然后后面三层依次以 2 的倍数减少，即 512、256 和 128 个神经元。后面的层级的神经元依次减少就压缩了前面层级中抽取的特征。当然，我们还能使用其它神经网络架构和神经元配置以更好地处理数据，例如卷积神经网络架构适合处理图像数据、循环神经网络适合处理时序数据，但本文只是为入门者简要地介绍如何使用全连接网络处理时序数据，所以那些复杂的架构本文并不会讨论。

```
# Model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

```

理解输入层、隐藏层和输出层之间变量的维度变换对于理解整个网络是十分重要的。作为多层感知机的一个经验性法则，后面层级的第一个维度对应于前面层级权重变量的第二个维度。这可能听起来比较复杂，但实际上只是将每一层的输出作为输入传递给下一层。偏置项的维度等于当前层级权重的第二个维度，也等于该层中的神经元数量。

**设计神经网络的架构**

在定义完神经网络所需要的权重矩阵与偏置项向量后，我们需要指定神经网络的拓扑结构或网络架构。因此占位符（数据）和变量（权重和偏置项）需要组合成一个连续的矩阵乘法系统。

此外，网络隐藏层中的每一个神经元还需要有激活函数进行非线性转换。激活函数是网络体系结构非常重要的组成部分，因为它们将非线性引入了系统。目前有非常多的激活函数，其中最常见的就是线性修正单元 ReLU 激活函数，本模型也将使用该激活函数。

```
# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

```

下图将展示本文构建的神经网络架构，该模型主要由三个构建块组成，即输入层、隐藏层和输出层。这种架构被称为前馈网络或全连接网络，前馈表示输入的批量数据只会从左向右流动，其它如循环神经网络等架构也允许数据向后流动。

![](https://segmentfault.com/img/remote/1460000012074739)

_前馈网络的核心架构_

**损失函数**

该网络的损失函数主要是用于生成网络预测与实际观察到的训练目标之间的偏差值。对回归问题而言，均方误差（MSE）函数最为常用。MSE 计算预测值与目标值之间的平均平方误差。

```
# Cost function

mse = tf.reduce_mean(tf.squared_difference(out, Y))

```

然而，MSE 的特性在常见的优化问题上很有优势。

**优化器**

优化器处理的是训练过程中用于适应网络权重和偏差变量的必要计算。这些计算调用梯度计算结果，指示训练过程中，权重和偏差需要改变的方向，从而最小化网络的代价函数。稳定、快速的优化器的开发，一直是神经网络和深度学习领域的重要研究。

```
# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

```

以上是用到了 Adam 优化器，是目前深度学习中的默认优化器。Adam 表示适应性矩估计，可被当作 AdaGrad 和 RMSProp 这两个优化器的结合。

**初始化器**

初始化器被用于在训练之前初始化网络的变量。因为神经网络是使用数值优化技术训练的，优化问题的起点是找到好的解决方案的重点。TensorFlow 中有不同的初始化器，每个都有不同的初始化方法。在这篇文章中，我使用的是 tf.variance_scaling_initializer()，是一种默认的初始化策略。

```
# Initializers
sigma = 
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

```

注意，用 TensorFlow 的计算图可以对不同的变量定义多个初始化函数。然而，在大多数情况下，一个统一的初始化函数就够了。

**拟合神经网络**

完成对网络的占位符、变量、初始化器、代价函数和优化器的定义之后，就可以开始训练模型了，通常会使用小批量训练方法。在小批量训练过程中，会从训练数据随机提取数量为 n=batch_size 的数据样本馈送到网络中。训练数据集将分成 n/batch_size 个批量按顺序馈送到网络中。此时占位符 X 和 Y 开始起作用，它们保存输入数据和目标数据，并在网络中分别表示成输入和目标。

X 的一个批量数据会在网络中向前流动直到到达输出层。在输出层，TensorFlow 将会比较当前批量的模型预测和实际观察目标 Y。然后，TensorFlow 会进行优化，使用选择的学习方案更新网络的参数。更新完权重和偏差之后，下一个批量被采样并重复以上过程。这个过程将一直进行，直到所有的批量都被馈送到网络中去，即完成了一个 epoch。

当训练达到了 epoch 的最大值或其它的用户自定义的停止标准的时候，网络的训练就会停止。

```
# Run initializer
net.run(tf.global_variables_initializer())
# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()
# Number of epochs and batch size
epochs = 10
batch_size = 256for e in range(epochs):
# Shuffle training data
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]
# Minibatch training
for i in range(0, len(y_train) // batch_size):
start = i * batch_size
batch_x = X_train[start:start + batch_size]
batch_y = y_train[start:start + batch_size]
# Run optimizer with batch
net.run(opt, feed_dict={X: batch_x, Y: batch_y})
# Show progress
if np.mod(i, 5) == 0:
# Prediction
pred = net.run(out, feed_dict={X: X_test})
line2.set_ydata(pred)
plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
plt.savefig(file_name)
plt.pause(0.01)

```

在训练过程中，我们在测试集（没有被网络学习过的数据）上评估了网络的预测能力，每训练 5 个 batch 进行一次，并展示结果。此外，这些图像将被导出到磁盘并组合成一个训练过程的视频动画。模型能迅速学习到测试数据中的时间序列的位置和形状，并在经过几个 epoch 的训练之后生成准确的预测。太棒了！

可以看到，网络迅速地适应了时间序列的基本形状，并能继续学习数据的更精细的模式。这归功于 Adam 学习方案，它能在模型训练过程中降低学习率，以避免错过最小值。经过 10 个 epoch 之后，我们完美地拟合了测试数据！最后的测试 MSE 等于 0.00078，这非常低，因为目标被缩放过。测试集的预测的平均百分误差率等于 5.31%，这是很不错的结果。

![](https://segmentfault.com/img/remote/1460000012074740)

_预测和实际 S&P 价格的散点图（已缩放）_

请注意其实还有很多种方法能进一步优化这个结果：层和神经元的设计、不同的初始化和激活方案的选择、引入神经元的 dropout 层、早期停止法的应用，等等。此外，其它不同类型的深度学习模型，比如循环神经网络也许能在这个任务中达到更好的结果。不过，这在我们的讨论范围之外。

**结论和展望**

TensorFlow 的发布是深度学习研究的里程碑事件，其高度的灵活性和强大的性能使研究者能开发所有种类的复杂神经网络架构以及其它机器学习算法。然而，相比使用高级 API 如 Keras 或 MxNet，灵活性的代价是更长的建模时间。尽管如此，我相信 TensorFlow 将继续发展，并成为神经网路和和深度学习开发的研究和实际应用的现实标准。我们很多客户都已经在使用 TensorFlow，或正在开发应用 TensorFlow 模型的项目。我们的 STATWORX 的数据科学顾问（https://www.statworx.com/de/data-science/）基本都是用 TensorFlow 研究课开发深度学习以及神经网络。

谷歌未来针对 TensorFlow 的计划会是什么呢？至少在我看来，TensorFlow 缺少一个简洁的图形用户界面，用于在 TensorFlow 后端设计和开发神经网络架构。也许这就是谷歌未来的一个目标:)![](https://segmentfault.com/img/remote/1460000012074741)

_原文链接：https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877_