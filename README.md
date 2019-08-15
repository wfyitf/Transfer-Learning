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

<div  align="center"> 

![](http://latex2png.com/pngs/cb121cb0b877b22faa4a8fd084467290.png)

</div>

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

### Domain Adaptation via Transfer Component Analysis(1) & Transfer Learning via Dimensionality Reduction(2)

Conference 1: Proceedings of the Twenty-First International Joint Conference on Artificial Intelligence (IJCAI-09)

Conference 2: Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence (2008) (AAAI-08)

Bibcitation1: @article{pan2010domain,
  title={Domain adaptation via transfer component analysis},
  author={Pan, Sinno Jialin and Tsang, Ivor W and Kwok, James T and Yang, Qiang},
  journal={IEEE Transactions on Neural Networks},
  volume={22},
  number={2},
  pages={199--210},
  year={2010},
  publisher={IEEE}
}

Bibcitation2: @inproceedings{pan2008transfer,
  title={Transfer learning via dimensionality reduction.},
  author={Pan, Sinno Jialin and Kwok, James T and Yang, Qiang and others},
  booktitle={AAAI},
  volume={8},
  pages={677--682},
  year={2008}
}

#### 内容

今天介绍的是两篇在迁移学习当中非常重要以及非常优秀的成果，TCA与JDA。首先TCA是一种unsupervised learning，相关的两篇论文的主要思想是**通过学习一个高维映射(kernel)，在此空间中源域与目标域的概率分布与标签的条件边缘保持近似（$p(X_S) \approx p(X_T)$，$p(y|X_S) \approx p(y|X_T)$**，并且在此条件下去最大化共同潜在域中的方差，以维持原有数据的特征属性（此限制具体参考
[MVU](http://new.aaai.org/Papers/AAAI/2006/AAAI06-280.pdf)）。具体的步骤如下：

- 1 首先在RKHS中定义如下距离(Maximum Mean Discrepancy)：
$$\operatorname{Dist}(\mathrm{X}, \mathrm{Y})=\left\|\frac{1}{n_{1}} \sum_{i=1}^{n_{1}} \phi\left(x_{i}\right)-\frac{1}{n_{2}} \sum_{i=1}^{n_{2}} \phi\left(y_{i}\right)\right\|_{\mathcal{H}}$$
将上式替换成源域与目标域之间的关系，可得：
$$\operatorname{Dist}\left(X_{S}^{\prime}, X_{T}^{\prime}\right)=\left\|\frac{1}{n_{1}} \sum_{i=1}^{n_{1}} \phi\left(x_{S_{i}}\right)-\frac{1}{n_{2}} \sum_{i=1}^{n_{2}} \phi\left(x_{T_{i}}\right)\right\|_{\mathcal{H}}^{2}$$

- 2 目标上式最小化，构造核矩阵$K$与$L$，如下图所示，图中有一些typo，比如列核矩阵应该有转置符号，不然维度会出现不对称：
![](https://cdn.mathpix.com/snip/images/f4ewSgOVoLXfm9cdN5HP31bCvxucTpcU694CkBFwqqQ.original.fullsize.png)（参考:[MMD推导](https://zhuanlan.zhihu.com/p/63026435)）
在原文中，$K=\left[\begin{array}{ll}{K_{S, S}} & {K_{S, T}} \\ {K_{T, S}} & {K_{T, T}}\end{array}\right]$, 
$L_{i j}=\left\{\begin{array}{ll}{\frac{1}{n_{1}^{2}}} & {x_{i}, x_{j} \in X_{s r c}} \\ {\frac{1}{n_{2}^{2}}} & {x_{i}, x_{j} \in X_{t a r}} \\ {-\frac{1}{n_{1} n_{2}}} & {\text { otherwise }}\end{array}\right.$

- 3 有了目标所需要的迹优化，根据MVU的启发，为保持数据散度/方差（距离）的一致性，需要最大化$trace(K)$，即最小化$-trace(K)$，再添加一个超参数，整个优化问题如下：
\begin{array}{cl}{\min _{K=\tilde{K}+\varepsilon I}} & {\operatorname{trace}(K L)-\lambda \operatorname{trace}(K)} \\ {\text { s.t. }} & {K_{i i}+K_{j j}-2 K_{i j}=d_{i j}^{2}, \forall(i, j) \in \mathcal{N}} \\ {} & {K \mathbf{1}=\mathbf{0}, \widetilde{K} \succeq 0}\end{array}

- 4 以上的问题可写为半定规划（SDP）来求得最优$K$，在有了$K$之后，可对$K$使用PCA来降维至$m\times(n_1+n_2)$维，对应的分别前$n_1$列即为源数据，后$n_2$列即为目标数据。然而此种方法无法泛化，不可对新的未知数据进行分类，所以作者在现有的核函数上通过降维构造一个$W$矩阵，来构造理想中的核矩阵（根据empirical kernel map）：
$$\widetilde{K}=\left(K K^{-1 / 2} \widetilde{W}\right)\left(\widetilde{W}^{\top} K^{-1 / 2} K\right)=K W W^{\top} K$$
其中$W=K^{-1 / 2} \widetilde{W} \in \mathbb{R}\left(n_{1}+n_{2}\right) \times m$为所需要求得映射矩阵。那么当有新的观测值出现时，可通过$\widetilde{k}\left(x_{i}, x_{j}\right)=k_{x_{i}}^{\top} W W^{\top} k_{x_{j}}$去映射至核矩阵。

- 5 最终问题变为以下的问题：
\begin{array}{cl}{\min _{W}} & {\operatorname{tr}\left(W^{\top} W\right)+\mu \operatorname{tr}\left(W^{\top} K L K W\right)} \\ {\text { s.t. }} & {W^{\top} K H K W=I}\end{array}
第一项正则是为了防止出现零解与奇异解，第二项则是维持降维变换$W^TK$的散度。确保数据再映射后结构不变。上式可通过构造拉格朗日算子并求导得到以下的优化问题：
\begin{array}{c}{\min _{W} \operatorname{tr}\left(\left(W^{\top} K H K W\right)^{\dagger} W^{\uparrow}(I+\mu K L K) W\right)} \\ 
or \quad {\max _{W} \operatorname{tr}\left(\left(W^{\top}(I+\mu K L K) W\right)^{-1} W^{\top} K H K W\right)}\end{array}
那么问题就转换为**广义瑞丽商**的求解。$W$可通过求得$(I+\mu K L K)^{-1} K H K$的前m个特征矢量拼合而成。这就完成了类似在4步骤中的PCA降维。在$W^TK$的空间中使用kNN分类器即可进行标准模式下的机器学习。

总结来说，TCA首先将原始数据映射至RKHS来构造一个较为高维($n_1+n_2 \times n_1+n_2$)的核函数矩阵（以数据之间的内积形式的表示，这里我感觉有一些奇怪，可能是因为这个内积把target与source给关联起来，以达到最小距离的目的），然后再通过构造一个低维度的矩阵，将这个已经关联的数据降维至m，并保持数据之间的方差（结构），以此来得到较好的效果。与8-13 Paper2不同的是，这篇文章在降维时同时考虑到了矩阵结构的保持。


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

JDA的工作是TCA的延伸，在TCA当中，假设了在$P(\mathcal{Y}|X_S) \approx (\mathcal{Y}|X_T)$, 然而在现实当中，这样的条件概率发生的情况很少，需要考虑到这个差异。作者在此基础上首先在源域上训练了一个分类器，然后通过建立在目标域的伪标签（pseudo label），并通过不断迭代来减少条件概率的差异。源域数据$\in R^{m\times n_s}$，目标域数据$\in R^{m\times n_t}$，目标寻求变换$A^T \in R^{k\times m}$。其步骤如下：

- 1 首先在RKHS中定义如下距离(Maximum Mean Discrepancy)，此步骤与TCA类似：
$$\operatorname{Dist}(\mathrm{X}, \mathrm{Y})=\left\|\frac{1}{n_{1}} \sum_{i=1}^{n_{1}} \phi\left(x_{i}\right)-\frac{1}{n_{2}} \sum_{i=1}^{n_{2}} \phi\left(y_{i}\right)\right\|_{\mathcal{H}}$$
将上式替换成源域与目标域之间的关系，可得：
$$\operatorname{Dist}\left(X_{S}^{\prime}, X_{T}^{\prime}\right)=\left\|\frac{1}{n_{1}} \sum_{i=1}^{n_{1}} \phi\left(x_{S_{i}}\right)-\frac{1}{n_{2}} \sum_{i=1}^{n_{2}} \phi\left(x_{T_{i}}\right)\right\|_{\mathcal{H}}^{2}$$
在JDA中，这个映射函数可为一个线性映射函数$A^T$，即：
$$\left\|\frac{1}{n_{s}} \sum_{i=1}^{n_{s}} \mathbf{A}^{\mathrm{T}} \mathbf{x}_{i}-\frac{1}{n_{t}} \sum_{j=n_{s}+1}^{n_{s}+n_{t}} \mathbf{A}^{\mathrm{T}} \mathbf{x}_{j}\right\|^{2}=\operatorname{tr}\left(\mathbf{A}^{\mathrm{T}} \mathbf{X} \mathbf{M}_{0} \mathbf{X}^{\mathrm{T}} \mathbf{A}\right)$$

- 2 条件概率自适应，在源域的数据上学习分类器（SVM,TCA等均可）后，应用于目标域的数据并且标上标签，现在在源域与目标域都有标签的情况下，对于标签的每一类$c \in {1,\cdots,C}$，其条件概率的具体为：
$$\left\|\frac{1}{n_{s}^{(c)}} \sum_{\mathbf{x}_{i} \in \mathcal{D}_{s}^{(c)}} \mathbf{A}^{\mathrm{T}} \mathbf{x}_{i}-\frac{1}{n_{t}^{(c)}} \sum_{\mathbf{x}_{j} \in \mathcal{D}_{t}^{(c)}} \mathbf{A}^{\mathrm{T}} \mathbf{x}_{j}\right\|^{2}=\operatorname{tr}\left(\mathbf{A}^{\mathrm{T}} \mathbf{X} \mathbf{M}_{c} \mathbf{X}^{\mathrm{T}} \mathbf{A}\right)$$
其中$\mathcal{D}_{s}^{(c)}=\left\{\mathbf{x}_{i} : \mathbf{x}_{i} \in \mathcal{D}_{s} \wedge y\left(\mathbf{x}_{i}\right)=c\right\}$是一组标签为$c$的源域实例。$n_s^{(c)}$为在源域上c类实例的个数，其余notation与之类似。
其中$\mathbf{M}_c$为：
<img src="https://cdn.mathpix.com/snip/images/B8Iz9rzRcym9G9mZOb2UCjv-Tv4QzNGAY6SK0Z2kS3A.original.fullsize.png" width="400" hegiht="213" align=center />

- 3 有了以上每一类的误差值，优化问题便转化为：
$$\min _{\mathbf{A}^{\mathrm{T}} \mathbf{X} \mathbf{H} \mathbf{X}^{\mathrm{T}} \mathbf{A}=\mathbf{I}} \sum_{c=0}^{C} \operatorname{tr}\left(\mathbf{A}^{\mathrm{T}} \mathbf{X} \mathbf{M}_{c} \mathbf{X}^{\mathrm{T}} \mathbf{A}\right)+\lambda\|\mathbf{A}\|_{F}^{2}$$
这个问题包含了以下几个目标函数：
    - 源域与目标域数据本身分布之间的差异。
    - 对于标签的条件概率分布之间的差异。
    - 映射后的数据要尽量保持大的方差。即结构保持不变，同TCA。
   
- 4 同时作者也给出了核-JDA的方法，即将$A^T$替换为$K$,既得：
$$\min _{\mathbf{A}^{\mathrm{T}} \mathbf{K} \mathbf{H}^{\mathrm{T}} \mathbf{A}=\mathbf{I}} \sum_{c=0}^{C} \operatorname{tr}\left(\mathbf{A}^{\mathrm{T}} \mathbf{K} \mathbf{M}_{c} \mathbf{K}^{\mathrm{T}} \mathbf{A}\right)+\lambda\|\mathbf{A}\|_{F}^{2}$$

- 5 通过拉格朗日算子去解得以上的优化目标，可得：
\begin{aligned} L &=\operatorname{tr}\left(\mathbf{A}^{\mathrm{T}}\left(\mathbf{X} \sum_{c=0}^{C} \mathbf{M}_{c} \mathbf{X}^{\mathrm{T}}+\lambda \mathbf{I}\right) \mathbf{A}\right) \\ &+\operatorname{tr}\left(\left(\mathbf{I}-\mathbf{A}^{\mathrm{T}} \mathbf{X} \mathbf{H} \mathbf{X}^{\mathrm{T}} \mathbf{A}\right) \mathbf{\Phi}\right) \end{aligned}
通过对$A$求导，可得到一下的规范化的特征值分解：
$$ \left(\mathbf{X} \sum_{c=0}^{C} \mathbf{M}_{c} \mathbf{X}^{\mathrm{T}}+\lambda \mathbf{I}\right) \mathbf{A}=\mathbf{X} \mathbf{H} \mathbf{X}^{\mathrm{T}} \mathbf{A} \boldsymbol{\Phi} $$
即以下的目标函数：
$$\left(\mathbf{X} \mathbf{H} \mathbf{X}^{\mathrm{T}} \right)^{-1}\left(\mathbf{X} \sum_{c=0}^{C} \mathbf{M}_{c} \mathbf{X}^{\mathrm{T}}+\lambda \mathbf{I}\right) \mathbf{A}= \lambda \mathbf{A}  $$
选择矩阵$\left(\mathbf{X} \mathbf{H} \mathbf{X}^{\mathrm{T}} \right)^{-1}\left(\mathbf{X} \sum_{c=0}^{C} \mathbf{M}_{c} \mathbf{X}^{\mathrm{T}}+\lambda \mathbf{I}\right) \in R^{n \times n}$最小的$k$个特征值所构成的特征矩阵即为$A$。

- 6 再每次求得$A$后，不断迭代源域标签直至收敛即可，算法如下：
<img src="https://cdn.mathpix.com/snip/images/N-H5_O1o1juU-gCdWDZyF-aUD_YUFCkkVk6I15dEn0Q.original.fullsize.png" width="500" hegiht="313" align=center />
