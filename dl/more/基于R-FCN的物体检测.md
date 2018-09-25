# 基于R-FCN的物体检测

![](http://upload-images.jianshu.io/upload_images/145616-effe12559d486ea2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

*   文章地址：arXiv:1605.06409.  
    《[R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://link.jianshu.com?t=https://arxiv.org/abs/1605.06409)》
*   Github链接：[https://github.com/daijifeng001/r-fcn](https://link.jianshu.com?t=https://github.com/daijifeng001/r-fcn)

(转载请注明出处：[\[译\] 基于R-FCN的物体检测 (zhwhong)](https://www.jianshu.com/p/db1b74770e52) )

* * *

# 摘要

我们使用R-FCN（region-based, fully convolutional networks）进行精确和有效的物体检测。对比之前的区域检测（Fast/Faster R-CNN \[6 , 18\] 应用于每一个区域子网格要花费数百次），我们的区域检测是基于整幅图片的全卷积计算。为了达到这个目标，我们使用了一个“位敏得分地图”（position-sensitive score maps）来权衡在图像分类中的平移不变性和在物体检测中的平移变换性这样一种两难境地。我们的方法采用了全卷积图片分类主干部分，例如用于物体检测的最新的残差网络（Residual Networks） (ResNets)\[9\]。在PASCAL VOC（e.g.，83.6% mAP on the 2007 set） 数据集的实验上，我们使用了101层ResNet达到了很好的效果。同时，我们仅仅使用了170ms/每张图片，比Faster R-CNN匹配快了2.5~20倍左右。公开的代码可以在此网站中访问到：[https://github.com/daijifeng001/r-fcn](https://link.jianshu.com?t=https://github.com/daijifeng001/r-fcn)

# 简介

比较流行的关于物体检测的深度网络可以通过ROI pooling layer分成两个子网络：

> （1）a shared, “fully convolutional” subnetwork independent of RoIs.（独立于ROI的共享的、全卷积的子网络）  
> （2）an RoI-wise subnetwork that does not share computation.（不共享计算的ROI-wise子网）

这种分解来源于较早之前的分类框架，例如：AlexNet\[10\]和VGGNets\[23\]，他们由两个子网组成，一个是以spatial pooling layer结束的卷积子网，一个是若干个fully-connected layers。因此，spatial pooling layrer很自然的在本实验中被转变成了ROI pooling layer。

但是目前最好的图像分类网络，例如残差网络（ResNets）和GoogleNets都是用fully convolutional设计的。通过分析，使用所有的卷积层去构建一个进行物体检测的共享的卷积网络是一件十分自然的事，而不把ROI-wise 子网作为隐藏层（hidden layer）。然而，通过实证调查，这个天真的想法需要考虑到inferior detection accuracy（极差的检测精度）与网络的superior classification accuracy（较高的分类精度）不匹配的问题。为了解决这个问题，在残差网络（ResNet）\[9\]中， ROI pooling layer of the Faster R-CNN detector 插入了两组卷积层，他们创造了一个更深的ROI-wise子网来提高准确度，由于每个ROI的计算不共享，因此速度会比原来的要慢。

上述设计主要是为了解决图片分类的平移不变性与物体检测之间的平移变换性之间的矛盾。一方面，图像级别的分类任务侧重于平移不变性（在一幅图片中平移一个物体而不改变它的判别结果），因此深度全卷积网络结构很适合处理这类图片分类的问题。

图一

图二

另一方面，物体检测任务需要定义物体的具体位置，因此需要平移变换特性。为了解决这矛盾，在ResNet的检测方法中插入了ROI pooling layer到卷积层（this region-specific operation breaks down translation invariance,and the post-RoI convolutional layers are no longer translation-invariant when evaluated across different regions）。然而，这个设计牺牲了训练和测试的效率，因为它引入了大量的region-wise layers。

在本篇文献中，我们开发出了一个称之为R-FCN（Region-based Fully Convolutional Network）的框架来用于物体检测。我们的网络由共享的全卷积结构，就像FCN一样\[15\]。为了把平移变换特性融合进FCN中，我们创建了一个位敏得分地图（position-sensitive score maps）来编码位置信息，从而表征相关的空间位置。在FCN的顶层，我们附加了一个position-sensitive ROI pooling layer 来统领这些得分地图（score maps）的信息，这些得分地图不带任何权重层。整个结构是端对端（end-to-end）的学习。所有可学习的层在整幅图片中都是可卷积的并且可共享的，并且可以编码用于物体检测的空间信息。图 1说明了这个关键的思路（key idea）,图 2比较了区域检测的各种算法。

使用101层的Residual Net作为骨架，我们的R-FCN在PASCAL VOC 2007测试集上达到了83.6%的mAP，在2012测试集上达到了82.0%的mAP。同时，我们的结果实现了170ms/每张图片的速度，比Faster R-CNN+ResNet-101 \[9\] 快了2.5~20倍。这个实验结果说明了我们的方法成功的解决了基于全卷积网络的图像级别的分类问题中的平移不变性和平移变换性之间的矛盾，就像ResNet能够被有效的转换成全卷积物体检测器（fully convolutional object detectors.）一样。详细的代码参见：[https://github.com/daijifeng001/r-fcn](https://link.jianshu.com?t=https://github.com/daijifeng001/r-fcn)

# 方法

## Overview（概览）

对于下面的R-CNN，我们采用了两种流行的策略：region proposal和region classification.我们通过region proposal Network（RPN）来抽取候选区域，它自身是一个全卷积结构。接下来，我们在RPN和R-FCN中共享这些特性。图 3展示了整个系统的结构。

图三

考虑到proposal region， R-FCN结构用来将ROI分类为物体和背景。在R-FCN中，所有可学习的权重层都是可卷积的并且是在整幅图片中进行计算。最后一个卷积层产生一堆K2个position-sensitive score maps 针对于每一个物体类别。因此有k2(C+1)个通道输出层（C个物体目录项+1个背景）。这K2个得分地图由K×k个空间网格来描述相对位置。例如，对于K×k = 3×3，这9个得分地图将物体类别编码为9个例子。

R-FCN以position-sensitive ROI pooling layer作为结束层。他将最后一个卷积层的输出结果聚集起来，然后产生每一个ROI的得分记录。我们的position-sensitive RoI层产生的是selective pooling，并且k×k的每个条目仅仅聚集来自于k×k得分地图堆里面的一个得分地图。通过端对端的训练，ROI 层带领最后一层卷积层去学习特征化的position-sensitive score maps。图 1说明了这个过程。图 4、图 5是两个可视化的例子。本算法的具体细节介绍参见后面的条目。

图四

图五

## Backbone architecture（主干架构）

R-FCN算法是基于ResNet-101\[9\]的，虽然其他的深度学习网络可以应用。RstNet-101有100个带global average pooling的卷积层，有一个1000级的fc层（fully-connected）。我们去掉了global average pooling和fc layer，然后只使用卷积层来计算feature maps。We use the ResNet-101 released by the authors of \[9\], pre-trained on ImageNet \[20\]。在ResNet-101中，最后一个卷积块是2048-d（2048维度）我们附加了一个随机初始化的1024d的1×1的卷积层来降低维度（更精确的，我们增加了卷积层的深度），然后我们使用了k2(C+1)个通道的卷积层来产生得分地图，下面会有具体的介绍。

## Position-sensitive socre maps & position-sentitive RoI pooling

为了在每个RoI中编码位置信息，我们通过一个网格把每个RoI分成k×k个bins。对于w×h的RoI区域，每一个bin的大小≈w/k×h/k。在我们的方法中，最后一个卷积层有k2个得分地图组成。对于第i行第j列的bin（0≤i , j≤k-1）, 其得分地图的计算公式为：

在本篇论文中我们通过平均得分来对ROI区域进行投票，从而产生（C+1）维的向量。然后我们计算了每一个目录项的softmax响应：

在训练和评级的时候，他们被用来估计交叉熵损失（cross-entropy loss）。

进一步的，我们用相似的方法定位了边界框回归（bounding box regression）。在k2(C+1)个卷积层，我们附加了一个4k2个卷积层用于边界框回归。Position-sensitive RoI pooling 在4k2的map中表现出来，对于每一个RoI，产生一个4k2位的向量，然后被聚合为一个4维向量通过平均投票。这个4维用 t = (t\_x, t\_y, t\_w, t\_h) 参数化一个边框。

Position-sensitive score maps 有一部分的灵感来自于实例级的语义分割FCNs。进一步的，我们引入了position-sensitive RoI pooling layer 来统领物体检测得分地图的学习。在ROI层之后，没有可学习的层，从而来加快训练和测试（training and inference）。

## Training（训练）

在预先计算了region proposals，端对端地训练R-FCN结构是非常简单的。接下来，我们的损失函数（loss fuction）的定义由两部分组成：交叉熵损失（cross-entropy loss）和边界回归损失（box regression loss）：

上式中，C_是RoI的真实标签（C_ = 0表示的是背景）。

是用于分类的交叉熵损失函数（cross-entropy loss）

> L_reg 是边界回归损失函数（bounding box regression loss）  
> t* 表示真实边框。  
> λ 被初始化设置为1。

当RoI与实际边框的重叠部分至少有0.5，那么我们认为是positive examples，否则是negative example。

在训练时，我们的方法采用OHEM（online hard example mining）是非常容易的。每个RoI区域的计算是可忽略的，从而使得样例挖掘（example mining）近乎是cost-free的。假定每张图片有N个proposals，一个直接的方法是，我们计算所有N个proposals的损失。然后我们对所有ROI按照损失进行排序。然后挑选B个具有最高损失的ROI。Backpropagation（反向传播算法）是基于选择的样例来演算的。因为我们的每个ROI的计算都是近似可以忽略的，所以forward time基本上不会受N的影响。而OHEM Fast R-CNN可能是会花费双倍的时间。在下表中，我们提供了一个详细的时间数据。

我们使用了0.0005的权重衰减系数和0.9的动量。缺省情况下，我们使用单尺度（single-scale）的训练：图片被统一调整为600个像素的大小。每个GPU处理1张图片，选取B=128ROI来用于反向传播。我们使用了8个GPU来训练模型（所以有效的小批量（mini-batch）大小是8×）。对于20k个mini-batches我们使用了0.001的学习速率；而对于10k个mini-batches，我们使用了0.0001的学习速率。为了让R-FCN拥有带RPN的特征，我们采用了4步交替训练，在RPN和R-FCN中交替训练。

## Inference（推论）

特征地图(feature maps)在RPN和R-FCN中共享计算。然后RPN部分选择出了ROI,而R-FCN部分评估了catagogy-wise scores和regresses bounding boxes。在推论阶段，我们计算了300个ROI区域。结果通过non-maximum suppression(NMS)来进行后处理，使用了0.3IoU的阈值，作为standard practice。

## A trous and stride

我们的全卷积网络架构是在FCN的基础上进行修改的。特别的，我们将ResNet-101的stride从32像素减少到16像素，从而增加了得分地图数量。All layers before and on the conv4 stage \[9\] (stride=16) are unchanged; the stride=2 operations in the first conv5 block is modified to have stride=1, and all convolutional filters on the conv5 stage are modified by the “hole algorithm” \[15, 2\] (“Algorithme à trous” \[16\]) to compensate for the reduced stride. For fair comparisons, the RPN is computed on top of the conv4 stage (that are shared withR-FCN), as is the case in \[9\] with Faster R-CNN, so the RPN is not affected by the à trous trick. The following table shows the ablation results of R-FCN (k _ k = 7 _ 7, no hard example mining). The à trous trick improves mAP by 2.6 points.

## Visualization（可视化）

在图 4和图 5我们展示了通过R-FCN学习到的position-sensitive score maps。不同的特征地图标志了不同的特征相对位置信息。例如：“top-center-sensitive”score map对于那些top-center位置关系的物体显示了较高的分数。如果一个候选框与真实物体精确的重合了（图 4），那么大多数的k2个bins会被强烈的激活，然后会得到较高的分数。相反的，如果候选边框与真实物体并没有完全准确的重合（图 5）那么有一些k2bins不会被激活，从而导致得分很低。

# 相关工作

R-CNN已经说明了带深度网络的区域候选的有效性。R-CNN计算那些关于裁剪不正常的覆盖区域的卷积网络，并且计算在区域直接是不共享的。SPPnet，Fast R-CNN和Faster R-CNN是半卷积的（semi-convolutional），在卷积子网络中是计算共享的，在另一个子网络是各自计算独立的区域。

物体检测器可以被认为是全卷积模型。OverFeat \[21\] detects objects by sliding multi-scale windows on the shared convolutional feature maps。在某些情况下，可以将单精度的滑动窗口改造成一个单层的卷积层。在Faster R-CNN中的RPN组件是一个全卷积检测器，用来预测是一个关于多尺寸的参考边框的实际边框。原始的RPN是class-agnostic（class无关的）。但是对应的clss-specific是可应用的。

另一个用于物体检测的是fc layer（fully-connected）用来基于整幅图片的完整物体检测。

# 实验

# 总结与展望

我们提出的Region-based Fully Convolutional Networks是一个简单、精确、有效的用于物体检测的框架。我们的系统很自然的采用了state-of –the –art 图片分类骨架，就像基于全卷积的ResNets一样。我们的方法比Faster R-CNN更精确，并且在训练和预测上都比它要快。

我们特意使本篇论文中给出的R-FCN看起来简单。其实仍然存在一系列的FCNS的正交扩展用来进行语义分割，还有一些基于区域方法的扩展用来进行物体检测。很高兴我们的系统能够享受到这些成果。

# Reference