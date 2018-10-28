# 详解 BiLSTM 及代码实现

> 作者：哈工大SCIR硕士生[@吴洋](//www.zhihu.com/people/c64ded7a9663711a04ad5ee0aa4f4c33)
> 
> 代码：哈工大SCIR博士生袁建华  
> 
> 来源：哈工大SCIR

## **一、介绍**

## **1.1 文章组织**

本文简要介绍了BiLSTM的基本原理，并以句子级情感分类任务为例介绍为什么需要使用LSTM或BiLSTM进行建模。在文章的最后，我们给出在PyTorch下BiLSTM的实现代码，供读者参考。

## **1.2 情感分类任务**

自然语言处理中情感分类任务是对给定文本进行情感倾向分类的任务，粗略来看可以认为其是分类任务中的一类。对于情感分类任务，目前通常的做法是先对词或者短语进行表示，再通过某种组合方式把句子中词的表示组合成句子的表示。最后，利用句子的表示对句子进行情感分类。

  

举一个对句子进行褒贬二分类的例子。

> 句子：我爱赛尔  
>   
> 情感标签：褒义

## **1.3 什么是LSTM和BiLSTM？**

LSTM的全称是Long Short-Term Memory，它是RNN（Recurrent Neural Network）的一种。LSTM由于其设计的特点，非常适合用于对时序数据的建模，如文本数据。BiLSTM是Bi-directional Long Short-Term Memory的缩写，是由前向LSTM与后向LSTM组合而成。两者在自然语言处理任务中都常被用来建模上下文信息。

## **1.4 为什么使用LSTM与BiLSTM？**

将词的表示组合成句子的表示，可以采用相加的方法，即将所有词的表示进行加和，或者取平均等方法，但是这些方法没有考虑到词语在句子中前后顺序。如句子“我不觉得他好”。“不”字是对后面“好”的否定，即该句子的情感极性是贬义。使用LSTM模型可以更好的捕捉到较长距离的依赖关系。因为LSTM通过训练过程可以学到记忆哪些信息和遗忘哪些信息。

  

但是利用LSTM对句子进行建模还存在一个问题：无法编码从后到前的信息。在更细粒度的分类时，如对于强程度的褒义、弱程度的褒义、中性、弱程度的贬义、强程度的贬义的五分类任务需要注意情感词、程度词、否定词之间的交互。举一个例子，“这个餐厅脏得不行，没有隔壁好”，这里的“不行”是对“脏”的程度的一种修饰，通过BiLSTM可以更好的捕捉双向的语义依赖。

## **二、BiLSTM原理简介**

## **2.1 LSTM介绍**

## **2.1.1 总体框架**

![](https://pic3.zhimg.com/v2-c8d4286fee2bccce9154e93819c8426c_b.jpg)

![](https://pic3.zhimg.com/80/v2-c8d4286fee2bccce9154e93819c8426c_hd.jpg)

总体框架如图1所示。

  

  

![](https://pic1.zhimg.com/v2-35ad506e9db04ca903d41997dcb677df_b.jpg)

![](https://pic1.zhimg.com/80/v2-35ad506e9db04ca903d41997dcb677df_hd.jpg)

图1. LSTM总体框架

## **2.1.2 详细介绍计算过程**

计算遗忘门，选择要遗忘的信息。

![](https://pic4.zhimg.com/v2-672731092e833db2e21ffc4014411e2c_b.jpg)

![](https://pic4.zhimg.com/80/v2-672731092e833db2e21ffc4014411e2c_hd.jpg)

  

![](https://pic1.zhimg.com/v2-7973a6bcc9257e45db952410538695d8_b.jpg)

![](https://pic1.zhimg.com/80/v2-7973a6bcc9257e45db952410538695d8_hd.jpg)

图2. 计算遗忘门

计算记忆门，选择要记忆的信息。

![](https://pic2.zhimg.com/v2-8dada7ceb4c750847cd3f8319f817c98_b.jpg)

  

![](https://pic3.zhimg.com/v2-54f7c8f29d17f30dd4d4a881e7b3910b_b.jpg)

![](https://pic3.zhimg.com/80/v2-54f7c8f29d17f30dd4d4a881e7b3910b_hd.jpg)

图3. 计算记忆门和临时细胞状态

  

计算当前时刻细胞状态

![](https://pic4.zhimg.com/v2-4ccbbb9016250ba63705cd834e0d5c0d_b.jpg)

![](https://pic4.zhimg.com/80/v2-4ccbbb9016250ba63705cd834e0d5c0d_hd.jpg)

  

![](https://pic4.zhimg.com/v2-7d88049a24256fecb5b61801072c6629_b.jpg)

![](https://pic4.zhimg.com/80/v2-7d88049a24256fecb5b61801072c6629_hd.jpg)

图4. 计算当前时刻细胞状态

计算输出门和当前时刻隐层状态

![](https://pic4.zhimg.com/v2-c457e33527e9d0807a0bb69f4a81c7c6_b.jpg)

![](https://pic4.zhimg.com/80/v2-c457e33527e9d0807a0bb69f4a81c7c6_hd.jpg)

  

![](https://pic4.zhimg.com/v2-ca64151ac1c6b658ac94eaca702c4e45_b.jpg)

图5. 计算输出门和当前时刻隐层状态

  

最终，我们可以得到与句子长度相同的隐层状态序列{ ![h_{0},h_{1}....,h_{n-1}](https://www.zhihu.com/equation?tex=h_%7B0%7D%2Ch_%7B1%7D....%2Ch_%7Bn-1%7D) }

## **2.2 BiLSTM介绍**

前向的LSTM与后向的LSTM结合成BiLSTM。比如，我们对“我爱中国”这句话进行编码，模型如图6所示。

  

![](https://pic4.zhimg.com/v2-bf3038dca90a59eb042ea767f684ed29_b.jpg)

![](https://pic4.zhimg.com/80/v2-bf3038dca90a59eb042ea767f684ed29_hd.jpg)

图6. 双向LSTM编码句子

![](https://pic2.zhimg.com/v2-af41362bbee404ccdbf3cd7634627942_b.jpg)

![](https://pic2.zhimg.com/80/v2-af41362bbee404ccdbf3cd7634627942_hd.jpg)

对于情感分类任务来说，我们采用的句子的表示往往是\[ ![h_{L2},h_{R2}](https://www.zhihu.com/equation?tex=h_%7BL2%7D%2Ch_%7BR2%7D) \]

。因为其包含了前向与后向的所有信息，如图7所示。

  

![](https://pic1.zhimg.com/v2-ed252ea820be0eac169f984dc4eaf924_b.jpg)

![](https://pic1.zhimg.com/80/v2-ed252ea820be0eac169f984dc4eaf924_hd.jpg)

图7. 拼接向量用于情感分类

## **三、BiLSTM代码实现样例**  
**3.1 模型搭建**

使用PyTorch搭建BiLSTM样例代码。代码地址为[https://github.com/albertwy/BiLSTM/](https://link.zhihu.com/?target=https%3A//github.com/albertwy/BiLSTM/)。

```py
#!/usr/bin/env python
# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(123456)


class BLSTM(nn.Module):
    """
        Implementation of BLSTM Concatenation for sentiment classification task
    """

    def __init__(self, embeddings, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(BLSTM, self).__init__()

        self.emb = nn.Embedding(num_embeddings=embeddings.size(0),
                                embedding_dim=embeddings.size(1),
                                padding_idx=0)
        self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # sen encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.output = nn.Linear(2 * self.hidden_dim, output_dim)

    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self, sen_batch, sen_lengths, sen_mask_matrix):
        """
        :param sen_batch: (batch, sen_length), tensor for sentence sequence
        :param sen_lengths:
        :param sen_mask_matrix:
        :return:
        """

        ''' Embedding Layer | Padding | Sequence_length 40'''
        sen_batch = self.emb(sen_batch)

        batch_size = len(sen_batch)

        ''' Bi-LSTM Computation '''
        sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)

        ''' Fetch the truly last hidden layer of both sides
        '''
        sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)  # (batch_size, 2*hid)

        representation = sentence_batch
        out = self.output(representation)
        out_prob = F.softmax(out.view(batch_size, -1))

        return out_prob

```

\_\_init\_\_()函数中对网络进行初始化，设定词向量维度，前向/后向LSTM中隐层向量的维度，还有要分类的类别数等。

  

bi\_fetch()函数的作用是将 ![h_{L2}](https://www.zhihu.com/equation?tex=h_%7BL2%7D) 与 ![h_{R2}](https://www.zhihu.com/equation?tex=h_%7BR2%7D) 拼接起来并返回拼接后的向量。由于使用了batch，所以需要使用句子长度用来定位开始padding时前一个时刻的输出的隐层向量。

  

forward()函数里进行前向计算，得到各个类别的概率值。

**3.2 模型训练**

```py
def train(model, training_data, args, optimizer, criterion):
    model.train()

    batch_size = args.batch_size

    sentences, sentences_seqlen, sentences_mask, labels = training_data

    # print batch_size, len(sentences), len(labels)

    assert batch_size == len(sentences) == len(labels)

    ''' Prepare data and prediction'''
    sentences_, sentences_seqlen_, sentences_mask_ = \
        var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)
    labels_ = Variable(torch.LongTensor(labels))
    if args.cuda:
        labels_ = labels_.cuda()

    assert len(sentences) == len(labels)

    model.zero_grad()
    probs = model(sentences_, sentences_seqlen_, sentences_mask_)
    loss = criterion(probs.view(len(labels_), -1), labels_)

    loss.backward()
    optimizer.step()

```

代码中training\_data是一个batch的数据，其中包括输入的句子sentences（句子中每个词以词下标表示），输入句子的长度sentences\_seqlen，输入的句子对应的情感类别labels。 训练模型前，先清空遗留的梯度值，再根据该batch数据计算出来的梯度进行更新模型。

```py
    model.zero_grad()
    probs = model(sentences_, sentences_seqlen_, sentences_mask_)
    loss = criterion(probs.view(len(labels_), -1), labels_)

    loss.backward()
    optimizer.step()

```

**3.3 模型测试**

以下是进行模型测试的代码。

```py
def test(model, dataset, args, data_part="test"):
    """
    :param model:
    :param args:
    :param dataset:
    :param data_part:
    :return:
    """

    tvt_set = dataset[data_part]
    tvt_set = yutils.YDataset(tvt_set["xIndexes"],
                              tvt_set["yLabels"],
                              to_pad=True, max_len=args.sen_max_len)

    test_set = tvt_set
    sentences, sentences_seqlen, sentences_mask, labels = test_set.next_batch(len(test_set))

    assert len(test_set) == len(sentences) == len(labels)

    tic = time.time()

    model.eval()
    ''' Prepare data and prediction'''
    batch_size = len(sentences)
    sentences_, sentences_seqlen_, sentences_mask_ = \
        var_batch(args, batch_size, sentences, sentences_seqlen, sentences_mask)

    probs = model(sentences_, sentences_seqlen_, sentences_mask_)

    _, pred = torch.max(probs, dim=1)

    if args.cuda:
        pred = pred.view(-1).cpu().data.numpy()
    else:
        pred = pred.view(-1).data.numpy()

    tit = time.time() - tic
    print "  Predicting {:d} examples using {:5.4f} seconds".format(len(test_set), tit)

    labels = numpy.asarray(labels)
    ''' log and return prf scores '''
    accuracy = test_prf(pred, labels)

    return accuracy


def cal_prf(pred, right, gold, formation=True, metric_type=""):
    """
    :param pred: predicted labels
    :param right: predicting right labels
    :param gold: gold labels
    :param formation: whether format the float to 6 digits
    :param metric_type:
    :return: prf for each label
    """
    ''' Pred: [0, 2905, 0]  Right: [0, 2083, 0]  Gold: [370, 2083, 452] '''
    num_class = len(pred)
    precision = [0.0] * num_class
    recall = [0.0] * num_class
    f1_score = [0.0] * num_class

    for i in xrange(num_class):
        ''' cal precision for each class: right / predict '''
        precision[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]

        ''' cal recall for each class: right / gold '''
        recall[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]

        ''' cal recall for each class: 2 pr / (p+r) '''
        f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 \
            else 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        if formation:
            precision[i] = precision[i].__format__(".6f")
            recall[i] = recall[i].__format__(".6f")
            f1_score[i] = f1_score[i].__format__(".6f")

    ''' PRF for each label or PRF for all labels '''
    if metric_type == "macro":
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    elif metric_type == "micro":
        precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
        recall = 1.0 * sum(right) / sum(gold) if sum(recall) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

```

## **四、总结**

本文中，我们结合情感分类任务介绍了LSTM以及BiLSTM的基本原理，并给出一个BiLSTM样例代码。除了情感分类任务，LSTM与BiLSTM在自然语言处理领域的其它任务上也得到了广泛应用，如机器翻译任务中使用其进行源语言的编码和目标语言的解码，机器阅读理解任务中使用其对文章和问题的编码等。

## **五、参考资料**

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://link.zhihu.com/?target=http%3A//colah.github.io/posts/2015-08-Understanding-LSTMs/)
