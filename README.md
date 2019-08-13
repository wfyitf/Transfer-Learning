# Transfer-Learning Paper notes
This repository contains contents on transfer learning.

Please install [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related) to view the equations, expressions and Latex related etc..!

### A Hybrid Instance-based Transfer Learning Method 

基于实例的混合权重迁移学习

#### 内容

本文是基于实例(instance based)的迁移学习典例，应用于医疗的人脸识别与受伤预测当中，为一种**监督学习**，其来源于以下均值误差的表达，讲来自目标域的实例的误差值表示为来自目标域与源域的两部分：

$$
\mathbf{E}_{x \sim P_{T}}[\underbrace{\mathcal{L}(\mathcal{A}(x), y)}_{\epsilon(x)}] 
$$


上式子第一项为target importance, 第二项为source importance， 由于在应用中源域与目标域的实例均为有限值，目标函数可写成：

$$\Theta^{*}=\underset{\Theta}{\arg \min }\left(\frac{\alpha}{N_{T}} \sum_{i=1}^{N_{T}} \epsilon\left(x_{i}, \Theta\right)+\frac{1-\alpha}{N_{S}} \sum_{j=1}^{N_{S}} \epsilon\left(x_{j}, \Theta\right) w_{x_{j}}\right)$$

作者的创新点在于之前的文献在考虑$w_{x_j}$时**只估算数据分布之前的权重**，而不考虑任务带来的影响，所以作者将权重分为域与任务两部分：
$$w_x = w_{domain,x} + w_{task,x}$$

其中$w_{domain,x}$是通过构造分类器来鉴别实例是来自于源域或者目标域获得,即（$P_T(x)/P_S(x)$）。对于$w_{task,x}$, 作者使用了不确定性（uncertainty）来作为权重，比如，在二分类中，此不确定性是实例$x$对于任务预测边界(decision boundary)的误差，若预测准确，则此权重为正，若预测错误，则此权重为负。

所以这个源域权重分为两部分，第一个为域权重，**表明若源域的数据越像目标域，则权重越大，反之。** 第二个为任务权重， **表明若源域的数据对于目标预测准确，则权重为正，并且越准确此权重越大，反之则为负，越不准确此权重越小。** 对于权重的构造，相当于筛选了源域越像目标域的实例，并且筛选了对预测有帮助的实例。在直觉上来说也是非常有效的。

并且通过实验证明这种方法在人脸识别与伤情预测中均取得了很好的效果。
