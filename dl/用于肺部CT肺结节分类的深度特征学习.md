# 用于肺部CT肺结节分类的深度特征学习

![](http://upload-images.jianshu.io/upload_images/145616-cefdff7ad73132f6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

*   原文链接(IEEE)：《[Deep feature learning for pulmonary nodule classification in a lung CT](https://link.jianshu.com?t=http://ieeexplore.ieee.org/document/7457462/?arnumber=7457462)》

(转载请注明出处：[【译】用于肺部CT肺结节分类的深度特征学习 (zhwhong)](https://www.jianshu.com/p/593b9c86fc52))

# 摘要

在这篇论文，我们提出了一个重要的在肺部CT确定肺结节的方法。具体地说，我们设计了一个从固有的原始手工图片特征中提取抽象信息的审年度神经网络。然后我们将深度学习出来的表述和开始的原始图像特征整合到一个长的特征矢量中。通过使用这个联合特征矢量，我们训练了一个分类器，之前通过了t-test的特征选择。为了验证提出的方法是有效的，我们用我们的内部数据集做了实验。内部数据集包括3598个肺结节（恶性：178，良性：3420），它们由一个医生手动分割。在我们的实验中，我们最高达到了95.5%的准确率，94.4%的敏感率和AUC达i到了0.987，比与我们竞争的其他方法表现优良。

# I. 介绍

在世界范围内，肺癌死亡是最常见的导致死亡的原因之一\[1\]。大量的方法被尝试用来减低肺癌死亡率。我们知道，一旦癌症在早期被检测出来，那么治疗会更有效也更有利于我们去克服它。此外，为了减轻医生们由于阅读大量CT而疲劳造成的误诊，计算机辅助检测引起了人们很大的兴趣。  
　　从临床角度来看，大于3mm的结节一般被称为肺结节\[2\]而更大的结节很容易变成癌细胞。因此，通过检测和观察结节的诊断筛选是重要的。为了这一目的，计算机辅助筛选系统在过去十年被提出，尽管由于它们的低性能而没有被用于临床。  
　　最近，受到深度学习在计算机视觉和语言识别领域的巨大成功的激励，很多人努力将这项技术用于医疗检测，特别是CT中的结节检测。比如，Roth等人用卷积神经网络（CNN）\[3\]，进行结节检测\[4\]。Ciompi等人，用CNN进行结节提取用来识别肺部围裂结节\[5\]。为了提高检测准确率，他们同时利用了轴、冠状面、矢状位面的信息。同时，Fakoor等人和Kumar等人，独立研究通过Stacked AutoEncoder（SAE）\[6\]进行心脏基因组分类\[6\]和肺结节分类\[7\]。尽管前述的基于深度学习的方法在他们自己的实验中也展现了很多成效，但他们大多忽略了如周长、圆周、集成密度、中值、偏度、峰值和结节这样的形态信息，这些信息并不能从卷积深度模型中提取出来。在这篇论文，我们提出了用深度模型来寻找潜在的形态特征，然后将深度学习到的信息和原始形态特征相结合。至于深度特征学习，我们是呀那个来Stacked Denosing AutoEncoder（SDAE）\[8\]。我们的工作受到了Suk等人工作\[11\]的启发，他们将阿尔茨海默氏病的原始的神经学影像特征和深度学习特征联系到了一起。

# II.提出的方法

### A.数据集和形态特征

我们收集了20个病人的CT扫描（男/女：7/13，年龄：63.5+-7.7）.肺结节由一个经验丰富的医生手工分割。总体上，我们有178个恶性和3420个良性结节（Table 1）.Figure 1给出了肺结节采样的样例，它们内部和之间变化很大，给结节分类带来了挑战。从每个结节我们提取了96个形态特征，即 area, mean Hounsfield Units (HU) 1 , standard deviation,mode, min, max, perimeter, circularity diameter, integrateddensity, median, skewness, kurtosis, raw integrated density,ferret angle, min ferret, aspect, ratio, roundness, solidity,entropy, run length matrix (44 values) \[9\], and gray-level cooccurrence matrix (32 values) \[10\].

### B.学习高度相关信息

为了更好的利用特征信息，我们用SDAE来发现形态特征之间潜在的非线性相关。SDAE的结构是按照等级划分的方式堆栈（stackong）多个自动编码器（autoencoder）。一个AE是一个有一个输入层一个隐藏层和一个输出层的多层神经网络。输入输出层神经元的个数由输入特征x∈R**d即d，而隐藏层神经元的数量可以是任意个。在AE中，隐藏层（h）和输出层（o）神经元的值如下得到：

其中Φ(.)是一个非线性sigmoid函数。其中W和b这样的参数通过不断学习这样隐藏层神经元可以覆盖输入特征的值，即x≈o。然而，为了使AE对于不希望的噪音更健壮，我们可以稍稍改动训练协议。实际上，在训练时我们通过增加随机噪音故意污染原始输入值，但是训练模型使输出层的值和原来的没有被污染的值接近。这种模型被称作“Denosing AutoEncoder”(DAE)\[8\]。[关于autoecnoder的原理](https://link.jianshu.com?t=http://blog.csdn.net/changyuanchn/article/details/15681853)  
　　注意隐藏层神经元的值可以被用作输入特征在新空间的另一表示，不同的唯独代表了对原始特征的不同联系。通过按等级堆栈许多DAE这样隐藏单元的值成了下一个更高AE的输入，我们建立了一个深度结构，我们称之为‘Stacked Denoising AutoEncoder’(SDAE)。  
　　DAE的一个显著优点是它的参数可以通过非监督的方式学习。所有我们可以利用尽可能多的训练实例而不管它们的标记信息是否被验证。这个有利的特征之后可以通过预训练的方法\[11\]寻找‘好’的SDAE初始值参数上被利用。  
　　简而言之，一个SDAE首先通过一个非监督的方式预训练然后预训练的参数值作为初始值来训练深度神经网络，通过在SDAE结构上多家一个标签层。然后我们通过一个监督方式微调所有的参数。在训练SDAE之后，我们去除最后的隐藏层的输出，即我们的SDAE标签层的输入，作为和固有的原始形态特征高度相关的值。我们最后将原始特征和SDAE学习到的特征通过一个长矢量联系在一起，将其作为我们新的增强特征矢量。

### C.特征选择和分类器训练

通过之前在模式识别领域的工作，我们很好地了解到在分类器选择千的特征选择对提升分类器性能是很有帮助的\[11\]。受他们的工作激励，我们应用了一个通过特征和种类标签之间的统计学测试进行特征选择的方法。实际上，我们对每个特征分别进行了一个简单的t-test，当测试的p-value大于预设门槛，我们认为对应的特征没有提供对分类有用的信息。基于被选择的特征，我们最后训练了一个线性的支持向量机（SVM），它以及在很多应用中证明了它作为一个分类器的效能\[12\]。

# III.实验结果

### A.实验设置

我们设计的SDAE有5层，其中有3层隐藏层。三层隐藏层的神经元数量是300,200,100. 对于SDAE训练，我们使用了一个大小为50的mini-batch [DeepLearnToolbox](https://link.jianshu.com?t=https://github.com/rasmusbergpalm/DeepLearnToolbox)。至于AE中的非线性sigmoid函数，我们使用了双曲正切函数(tanh)。为了最好利用我们的数据采样和DAE训练的非监督特性，我们使用了我们在预训练（迭代200次）数据集中的所有样例，即178个恶性结节和3420个良性结节。  
　　需要注意由于良性和恶性训练样例的不平衡，我们从3420个良性类型中随机挑选了200个。在我们的性能评估中，我们只利用178个恶性和200个良性（随机挑选）结节进行了五倍交叉验证。换一句话，我们将五分之一的样例放在一边只用作测试然后用剩下的五分之四样例。我们需要强调的是，在微调我们的SDAE和SVM学习中我们用的五分之四样例和留下的测试样例毫无关系。  
　　在微调SDAE之后，我们通过联合最后隐藏层的输出即100个值和原始96维特征得到了一个196维的增强特征矢量。在特征选择阶段，我们将p-value的门槛设置为0.001.最后，我们的SVM的模型超参数用libSVM库(Available at [https://www.csie.ntu.edu.tw/~cjlin/libsvm/](https://link.jianshu.com?t=https://www.csie.ntu.edu.tw/~cjlin/libsvm/))由一个在空间{2**-5, 2**-4,... ,2\*\*4,2\*\*5}的五倍交叉嵌套验证决定。为了验证提出的通过深度学习的特征进行的特征增强是效能，我们把我们的方法和一个只用了原始特征的卷积方法进行了比较。需要注意所有其他的对相比较的方法的设置和我们提出的方法是完全一样的。

### B.性能表现

我们使用了四个度量进行性能评测，即准确性，敏感性，特异性和接受者操作特意曲线下的区域（AUC）。Figure 1比较了用不同测试单元进行的性能评价AUC的平均值，而其他性能用百分号表示。Original+SDAE特征在每个性能表现都更优异。特别的，准确性和敏感性分别提高了2.1%和3.4%。

# IV.总结

在这篇论文，我们提出来用深度结构去寻找CT扫描的肺结节分类中潜在的非线性形态信息。临床上，在早起阶段找到恶性结节是十分重要的。我们的深度学习特征在分别良性和恶性结节方面表现了巨大威力，在敏感度方面有了巨大提高。

# Acknowledge

This research was supported by Basic Science Research Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Education (NRF-2015R1C1A1A01052216).The authors gratefully acknowledge technical supports from Biomedical Imaging Infrastructure, Department of Radiology, Asan Medical Center.

# Reference

\[1\] B. W. Stewart and C. P. Wild, editors, World Cancer Report 2014. Lyon,France: International Agency for Research on Cancer, Feb. 2014.

\[2\] M. K. Gould, J. Fletcher, M. D. Iannettoni, W. R. Lynch, D. E. Midthun,D. P. Naidich, and D. E. Ost, “Evaluation of Patients with Pulmonary Nodules: When is it lung cancer?: ACCP evidence-based clinical practice guidelines, 2ed edition” Chest, Vol. 132, No. 3, pp. 108-130, Sep. 2007.

\[3\] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based Learning Applied to Document Recognition,” Proceedings of the IEEE,Vol. 86, No. 11, pp. 2278-2324, Nov. 1998.

\[4\] H. R. Roth, L. Lu, J. Liu, J. Yao, A. Seff, K. Cherry, L. Kim, and R. M.Summers, “Improving Computer-aided Detection using Convolutional Neural Networks and Random View Aggregation,” arXiv:1505.03046,Sep. 2015.

\[5\] F. Ciompi, B. de Hoop, S. J. van Riel, K. Chung, E. Th. Scholten, M.Oudkerk, P. A de Jong, M. Prokop, and B. van Ginneken, “Automatic Classification of Pulmonary Peri-Fissural Nodules in Computed Tomography Using an Ensemble of 2D views and a Convolutional Neural Network Out-of-the-Box,” Medical Image Analysis, Vol. 26, No. 1, pp.195-202, Dec. 2015.

\[6\] R. Fakoor, F. Ladhak, A. Nazi, and M. Huber, “Using Deep Learning to Enhance Cancer Diagnosis and Classification,” Proc. of the ICML Workshop on the Role of Machine Learning in Transforming Healthcare (WHEALTH), Vol 28, June 2013

\[7\] D. Kumar, A. Wong, and D. A. Clausi, “Lung Nodule Classification Using Deep Features in CT Images,” Proc. 12th Conference on Computer and Robot Vision (CRV), pp. 133-138, June 2015.

\[8\] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P. Manzagol,“Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion,” The Journal of Machine Learning Research, Vol. 11, pp. 3371-3408, Mar. 2010.

\[9\] X. Tang, “Texture Information in Run-length Matrices,” IEEE Transactions on Image Processing, Vol. 7, No. 11, pp.1602-1609, Nov.1998.

\[10\] R. M. Hatalick, K. Shanmugam, and I. Dinstein, “Textural Features for Image Classificaiton,” IEEE Transactions on Systems, Man and Cybernetics, Vol. 3, No. 6, pp. 610-621, Nov. 1973.

\[11\] H.-I. Suk, S.-W. Lee, and D. Shen, “Latent Feature Representation with Stacked Auto-Encoder for AD/MCI Diagnosis,” Brain Structure & Function, Vol. 220, No. 2, pp. 841-859, Mar. 2015.

\[12\] G. E. Hinton and R. R. Salakhutdinov, “Reducing the Dimensionality of Data with Neural Networks,” Science, Vol. 313, No. 5786, pp. 504-507, July 2006.

\[13\] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning Representations by Back-propagating Errors,” Nature, Vol. 323, No. 9, pp.533-536, Oct. 1986.