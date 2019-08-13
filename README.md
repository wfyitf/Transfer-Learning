# Transfer-Learning Paper notes
This repository contains contents on transfer learning.

Please install [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related) to view the equations, expressions and Latex related etc..!

# 2019-08-13

## Paper 1

### A Hybrid Instance-based Transfer Learning Method 

Journal: Machine Learning for Health (ML4H) Workshop at NeurIPS 2018 (NIPS workshop)

Bibcitation:

@article{antropova2018machine,
  title={Machine Learning for Health (ML4H) Workshop at NeurIPS 2018},
  author={Antropova, Natalia and Bream, Andrew and Beaulieu-Jones, Brett K and Chen, Irene and Chivers, Corey and Dalca, Adrian and Finlayson, Sam and Fiterau, Madalina and Fries, Jason Alan and Ghassemi, Marzyeh and others},
  journal={arXiv preprint arXiv:1811.07216},
  year={2018}
}

基于实例的混合权重迁移学习

#### 内容

本文是基于实例(instance based)的迁移学习典例，应用于医疗的人脸识别与受伤预测当中，为一种**监督学习**，其来源于以下均值误差的表达，讲来自目标域的实例的误差值表示为来自目标域与源域的两部分：


![](http://latex2png.com/pngs/60dd81cb69610f718cdcbf8be180a001.png)
<!--
\begin{aligned} \mathbb{E}_{x \sim P_{T}}[\underbrace{\mathcal{L}(\mathcal{A}(x), y)}_{\epsilon(x)}] &=\int \epsilon(x) P_{T}(x) d x=\int \epsilon(x) \underbrace{\left[\alpha+(1-\alpha) \frac{P_{S}(x)}{P_{S}(x)}\right]}_{=1} P_{T}(x) d x \\ &=\alpha \mathbb{E}_{x \sim P_{T}}[\epsilon(x)]+(1-\alpha) \mathbb{E}_{x \sim P_{S}}\left[\epsilon(x) \frac{P_{T}(x)}{P_{S}(x)}\right] 
\end{aligned} -->

上式子第一项为target importance, 第二项为source importance， 由于在应用中源域与目标域的实例均为有限值，目标函数可写成：

$$\Theta^{*}=\underset{\Theta}{\arg \min }\left(\frac{\alpha}{N_{T}} \sum_{i=1}^{N_{T}} \epsilon\left(x_{i}, \Theta\right)+\frac{1-\alpha}{N_{S}} \sum_{j=1}^{N_{S}} \epsilon\left(x_{j}, \Theta\right) w_{x_{j}}\right)$$

作者的创新点在于之前的文献在考虑$w_{x_j}$时**只估算数据分布之前的权重**，而不考虑任务带来的影响，所以作者将权重分为域与任务两部分：
$$w_x = w_{domain,x} + w_{task,x}$$

其中$w_{domain,x}$是通过构造分类器来鉴别实例是来自于源域或者目标域获得,即（$P_T(x)/P_S(x)$）。对于$w_{task,x}$, 作者使用了不确定性（uncertainty）来作为权重，比如，在二分类中，此不确定性是实例$x$对于任务预测边界(decision boundary)的误差，若预测准确，则此权重为正，若预测错误，则此权重为负。

所以这个源域权重分为两部分，第一个为域权重，**表明若源域的数据越像目标域，则权重越大，反之。** 第二个为任务权重， **表明若源域的数据对于目标预测准确，则权重为正，并且越准确此权重越大，反之则为负，越不准确此权重越小。** 对于权重的构造，相当于筛选了源域越像目标域的实例，并且筛选了对预测有帮助的实例。在直觉上来说也是非常有效的。

并且通过实验证明这种方法在人脸识别与伤情预测中均取得了很好的效果。

## Paper 2
### Structural Domain Adaptation With Latent Graph Alignment
Conference: 2018 25th IEEE International Conference on Image Processing (ICIP)

Bibcitation:
@inproceedings{zhang2018structural,
  title={Structural Domain Adaptation with Latent Graph Alignment},
  author={Zhang, Yue and Miao, Shun and Liao, Rui},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={3753--3757},
  year={2018},
  organization={IEEE}
}

#### 内容

本文针对最小化**最大均值差异(Maximum Mean Discrepancy)** 提出了相对应的领域自适应方法，此种方法在TCA,JDA等方法的启发下，在RKHS中最小化MMD的同时通过构造Laplacian Graph进行图谱分析来最大化结构的相似度。文章分为两个步骤，一为通过寻找一对线性权重(源域权重与目标域权重)$W_s$与$W_t$来最小化**经验式**的MMD，具体形式如下：

$$\operatorname{MMD}\left(W_{s}, W_{t}\right)=\left\|\frac{1}{n_{s}} \sum_{i=1}^{n_{s}} W_{s}^{T} x_{i}-\frac{1}{n_{t}} \sum_{j=1}^{n_{t}} W_{t}^{T} \hat{x}_{j}\right\|$$

其中$W_s$与$W_t$为我们需要求得的映射，$n_s$为源域实例数目，$n_t$为目标域实例数目，$x_i$与$\hat{x}_j$分别为源域与目标域的实例。 同时在求得$W_s$与$W_t$后，通过构造拉普拉斯矩阵来衡量两个域(或可看成流行)的结构相似度。其损失函数可定义为：

$$\mathcal{L}\left(W_{s}, W_{t}\right)=\left\|\Lambda\left(U^{T} L_{s} U\right)-\Lambda\left(L_{t}\right)\right\|_{2}^{2}$$

其中$L_s$与$L_t$为源域与目标域正则化的拉普拉斯矩阵。所以迁移的问题便转化为在使得这个函数尽量小的情况下去最小化MMD。如下式所示：

![](http://latex2png.com/pngs/6eb43870aa801bfc313fe49b84da11f6.png)

然而对于求得最优$W_s$与$W_t$，这不是一个凸问题，作者采用迭代方法将问题分解为两个步骤：

- ###  1 MMD-Stage
在第一个步骤当中，问题转化为：在第$i$次迭代中，给定$W_t^i$，可解出$W_s^i$:

$$W_{s}^{i}=\underset{W}{\operatorname{argmin}} \operatorname{MMD}\left(W, W_{t}\right), \quad \text { s.t. } W_{t}=W_{t}^{i}$$

这是一个标准的MMD模型，存在闭合解，可参考TCA等文章求得近似解。

- ###  2 LGA-Stage
在第二个步骤当中，问题转化为：在第$i$次迭代中，给定$W_s^i$，可解出$W_{t+1}^i$:

$$ W_{t}^{i+1}=\underset{W}{\operatorname{argmin}} \mathcal{L}\left(W_{s}^{i}, W\right)$$

为解决谱优化不存在闭合解的问题，作者通过构造中间图后通过sgd(梯度下降)的方法来迭代求得$W_{t+1}$。其中拉普拉斯的图构造通过热核(heat kernel)权重进行构造。

其余具体的constraints与估计推导与concern可参考原文。

# 2019-08-14

## Paper 1

### Domain Adaptation via Transfer Component Analysis

Conference: Proceedings of the Twenty-First International Joint Conference on Artificial Intelligence (IJCAI-09)

Bibcitation:@article{pan2010domain,
  title={Domain adaptation via transfer component analysis},
  author={Pan, Sinno Jialin and Tsang, Ivor W and Kwok, James T and Yang, Qiang},
  journal={IEEE Transactions on Neural Networks},
  volume={22},
  number={2},
  pages={199--210},
  year={2010},
  publisher={IEEE}
}

#### 内容

今天介绍的是两篇在迁移学习当中非常重要以及非常优秀的成果，TCA与JDA。


## Paper 2

### Transfer feature learning with joint distribution adaptation

Conference: Proceedings of the IEEE international conference on computer vision (ICCV-2013)

Bibcitation:@inproceedings{long2013transfer,
  title={Transfer feature learning with joint distribution adaptation},
  author={Long, Mingsheng and Wang, Jianmin and Ding, Guiguang and Sun, Jiaguang and Yu, Philip S},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2200--2207},
  year={2013}
}

#### 内容
