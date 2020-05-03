---
layout: post
title:  "CS229 学习笔记"
---

# 吴恩达机器学习 学习笔记
https://www.bilibili.com/video/av50747658


## 用于公式引用
高斯分布概率密度函数：
$$
\begin{aligned}
P(x)=\frac{1}{\sqrt{2π}\sigma}exp({-\frac{(x-\mu)^2}{2\sigma^2}})\\
p(z)=\frac{1}{(2π)^{\frac{n}{2}}|\Sigma|^\frac{1}{2}}exp(-\frac{1}{2}
(x-\mu)^T\Sigma^{-1}(x-\mu))
\end{aligned}
$$

伯努利分布概率密度函数：

$$P(y)=\phi^y\cdot(1-\phi)^{(1-y)}$$


sigmoid 函数：

$$
g(z)=\frac{1}{1+e^{-z}}
$$





## P2
+ 什么是机器学习：程序在task T上的performance P随着experience E而增加


# 斯坦福cs229 机器学习课程（吴恩达）
https://www.bilibili.com/video/av79827258?from=search&seid=13988545588995524424
## p1
主要讲了五个概念：supervised learning, machine learning strategy, deep learning, unsupervised learning, reinforcement learning.

cs229a:更少数学，更偏应用

## p2
note:http://cs229.stanford.edu/notes2019fall/cs229-notes1.pdf
### linear regression
线性回归的hypothesis是模型是线性的：h(x)=h_0+h_1x 

损失函数 
$$
J(\theta)=\frac{1}{2}\sum (h_\theta(x^{(i)})-y^{(i)})^2
$$
其中二分之一是为了方便求导后运算，平方的原因将在正则化章节解释（下周）


### batch/stochastic gradient descent
在线性回归的梯度下降中，不用担心存在非全局最优的局部最优解。
因为它的损失函数是二次的。也可以通过等高线图来发现。

课程中colon equal := 相当于赋值
$$
\theta_j:=\theta_j-\alpha \cdot\frac{\partial J(\theta)}{\partial\theta_j}
$$
alpha表示学习率learning rate

把损失函数代入，再经过简单计算
$$
\theta_j:=\theta_j-\alpha \cdot (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

batch gradient descent：训练完一整批数据集后，作一次参数的更新，常用于小规模数据集。   
stochastic gradient descent：每训练一个数据，就作一次参数的更新，避免因为数据集过大导致训练时间很久。但是会导致解在全局最优解附近震荡，而不一定是正正好好全局最优解


当你的损失函数值随着训练却增加，说明学习率设置的过高。

### normal equation
对于线性回归问题，用normal equation方法不需要上述递归，就可以得到全局最优解

$\nabla_\theta J(\theta)$表示$J(\theta)$关于$\theta$的导数 其中$\theta$是n维向量

所以
$$
\nabla_\theta J(\theta) = 
\begin{bmatrix} 
\frac{\partial J(\theta)}{\partial\theta_0} \\\\ 
\frac{\partial J(\theta)}{\partial\theta_1} \\ ...\\\\
\frac{\partial J(\theta)}{\partial\theta_n} 
\end{bmatrix}
$$
由于模型是线性的，$J(\theta)$是二次的，因此令$\nabla_\theta J(\theta) = 0$，就可以得到极值，来得到全局最优解

##### 矩阵的迹 tr(A) 以及推导时用到的公式
对于方阵A，tr A 或者tr(A)表示A的对角元素之和

可以证明（课后）：若$f(A)=tr(A B^T)$，那么$\nabla_Af(A)=B^T$

$tr(AB) = tr(BA)$

$\nabla_Atr(AA^TC)=CA+C^TA$
此公式可以在一维中类比为$\frac{d}{da} a^2c=2ac$



证明：

##### 回到normal equation
将$J(\theta)=\frac{1}{2}\sum (h_\theta(x^{(i)})-y^{(i)})^2$用矩阵和向量表示（详细见视频71:45）

$$
J(\theta)=\frac{1}{2}(X\theta-y)^T(X\theta - y)
$$
其中$X=\begin{bmatrix} 
——(x^{(1)})^T—— \\\\ 
——(x^{(2)})^T—— \\ ...\\\\
——(x^{(m)})^T—— 
\end{bmatrix}$，
$\theta=\begin{bmatrix} 
\theta_0 \\\\ 
\theta_1 \\ ...\\\\
\theta_n
\end{bmatrix}$,
$y=\begin{bmatrix} 
y^{(1)} \\\\ 
y^{(2)} \\ ...\\\\
y^{(m)}
\end{bmatrix}$

展开后
$$
J(\theta)=\frac{1}{2}(X^T\theta^T-y^T)(X\theta - y)

=\frac{1}{2}(X^T\theta^TX\theta-X^T\theta^Ty-y^TX\theta+y^Ty)
$$
对$\theta$求导（详见http://cs229.stanford.edu/notes2019fall/cs229-notes1.pdf  中p9~10）得

$$
\nabla_\theta J(\theta) =X^TX\theta-X^Ty
$$
令其为零，得到$\theta$的normal equation
$$
\theta = (X^TX)^{-1}X^Ty
$$

## p3 线性代数复习 Linear Algebra Review
note:
http://cs229.stanford.edu/section/cs229-linalg.pdf

note 比 lecture 重要，lecture能帮助阅读note

### rank 秩
rank表示线性无关列/行的极大数目

### 矩阵乘向量 Mv 的两种理解
m行n列的矩阵M 乘 n维向量v
1. m次 两个n维向量的内积
2. 矩阵M对向量v施加坐标变换：单位基向量变换为n个m维向量

### 矩阵A(m x k)乘矩阵B(k x n) 的两种理解
1. m个k维横向量 与 n个k维纵向量 分别求内积，放到对应位置
2. k个m维纵向量 与 k个n维横向量 的外积（矩阵）之和：m维纵向量与n维横向量外积组成m x n的矩阵，共k个这样的矩阵相加

### 线性代数在机器学习中的作用
1. 用来表示数据：$X\in\mathbb{R}^{m\times n}$表示m个examples和n个features
2. 概率论中许多概念都用矩阵表示，例如协方差矩阵
3. 微积分/优化理论：例如雅可比矩阵（Jacobian）、海森矩阵（Hessian）
4. 核方法：核矩阵

### 将矩阵看作是一个函数（变换）：同YouTube视频
矩阵（变换）可逆的条件：方阵 & 满秩 

### 向量投影矩阵
$proj(\vec{b};\vec{v})=
\begin{bmatrix} 
\frac{\vec{v} \cdot \vec{v}^T}{\vec{v}^T \cdot \vec{v}}
\end{bmatrix} \cdot \vec{b}$

推导详见  https://blog.csdn.net/wlk1229/article/details/84779370

### 第二种线性回归的normal equation 的解法
$\theta$是n维向量，表示n个参数，X是m个example，n个feature的m x n的输入矩阵，y是吗、维输出向量。将X视为一个函数变换

$$
X:\mathbb{R^n}\rightarrow \mathbb{R^m}
$$

一般情况下m>n，因此所有$\theta$（n维空间）上的点经过X函数变换后投影在y（m维空间）中的一个子空间（subspace）上，这个子空间由X唯一确定了。如果向量y刚好在这个子空间上，那么理论上损失函数$J(\theta)$可以为0。

在线性回归问题中，我们要找到合适的向量$\theta$，使它经过X函数变换后最接近向量y。即
$$
X(\theta) = proj(y;X)
$$
根据上一章的向量投影矩阵公式
$$
X\cdot\theta = X(X^TX)^{-1}X^T\cdot y

\theta = (X^TX)^{-1}X^T\cdot y
$$
同p2中normal equation 解一致。

### decomposition 特征分解和奇异值分解
任何一个变换矩阵A，都可以分解(decomposition)为3个步骤：旋转rotation#1，拉伸scaling，再旋转 rotation#2

即 将一个函数分解为3个子函数，将一个矩阵分解为3个满足一定格式的矩阵

eigen decomposition特征分解 : rotation 1, scaling(Complex), rotation$^{-1}$(1的逆旋转)

SVD奇异值分解： rotation 1, scaling(Real), rotation 2

其中rotation矩阵满足正交矩阵， scaling矩阵满足对角矩阵

如果一个矩阵的特征分解和奇异值分解矩阵一样，那么它一定是对称矩阵。

所有特征值构成的集合称为频谱(spectrum)

一个矩阵特征值的总和等于它的迹trace，特征值的乘积等于它的行列式的值

## p4
http://cs229.stanford.edu/notes/cs229-notes1.pdf

outline：复习线性回归、locally weighted linear regression、 probabilistic interpretation、 逻辑回归、 牛顿法
### 线性回归复习
对于一些模型不是线性的问题，我们也可以使用技巧将它们转换为线性回归。

例如$h(\theta) = h_0+h_1x+h_2\sqrt{x}$，可以转换为$h(\theta) = h_0+h_1x_1+h_2x_2$

### LWLR(locally weighted linear regression)局部加权线性回归

'Parametric' learning algorithm: fit 固定的参数集合（$\theta_i$）to data， 例如线性回归

'Non-parametric' learning algorithm:参数的数量会根据数据量而改变。
缺点：它需要一直存储着之前训练完成的数据，因此不适合超大量数据集。优点：不需要人工提前猜测（设置）参数。

和线性回归的模型相同，具体方法与线性回归类似，但是***在每一次预测新样本时都会重新确定参数***。

训练的基本思想是：越靠近预测点的数据权重越大，越远离预测点的数据越不重要，因此在训练过程中每个数据都有权重w。基本上满足以上思想的函数都可以用来作为权重函数。

损失函数：

$$
J(\theta) = \sum_{i=1}^{m}\omega^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2

其中\omega^{(i)}=exp(-\frac{(x^{(i)}-x)^2}{2\tau^2}),\tau为bandwidth，定义靠近的带宽
$$

可参考note：https://blog.csdn.net/qq_31589695/article/details/79825189
### Probabilistic Interpretation 25:20

为什么使用最小二乘作为损失函数？

假设实际的y满足$y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}$其中 $\epsilon^{(i)}$~$N(0,\sigma^2)$
即$P(\epsilon^{(i)})=\frac{1}{\sqrt{2π}\sigma}exp({-\frac{(\epsilon^{(i)})^2}{2\sigma^2}})$且i.i.d(independent and identical distributed独立同分布)

$P(y^{(i)}|x^{(i)};\theta)$中的分号(semi colon)被读作parameterized by，因为$\theta$在这个情况下不是一个随机变量，而是一个确定的参数，不能用$P(y^{(i)}|x^{(i)},\theta)$

那么$(y^{(i)}|x^{(i)};\theta)$~$N(\theta^Tx^{(i)},\sigma^2)$
likelihood of $\theta$(似然函数)$L(\theta)$
$$
L(\theta)=P(Y|X;\theta)

=\prod_{i=1}^mP(y^{(i)}|x^{(i)};\theta)

=\prod_{i=1}^m\frac{1}{\sqrt{2π}\sigma}exp({-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}})
$$

在数值上参数$\theta$的似然函数 与 数据的概率相等，但是描述的对象是不同的。

为了便于计算，对似然函数取对数

$$
l(\theta) = logL(\theta)

=log\prod_{i=1}^m\frac{1}{\sqrt{2π}\sigma}exp({-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}})

=m\cdot log\frac{1}{\sqrt{2π}\sigma}+\sum_{i=1}^m{-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}}
$$

我们使用MLE(Maximum likelihood estimator)极大似然估计来选择参数$\theta$

要使$L(\theta)$最大，即要使$l(\theta)$最大，而$m和\sigma$都是常数，因此即要使$\frac{1}{2}\sum (y^{(i)}-\theta^Tx^{(i)})^2$最小，也就是线性回归中的损失函数$J(\theta)$

因此线性回归中损失函数选择最小二乘是基于数据（误差）满足正态分布的假设而来的

因此，基于概率统计，关键步骤是：
1. 做好假设，算出$P(Y|X;\theta)$
2. 极大似然估计

### 分类问题（二分类）　logistics regression
对于分类问题，我们的目标是让输出在[0,1]间，因此我们的hypothesis从线性回归时的$h_\theta(x)=\theta^Tx$ 变成逻辑回归中的
$$
h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$
其中$g(z)=\frac{1}{1+e^{-z}}$ 它被称为"sigmoid function"或"logistic function"

假设$P(y=1|x;\theta)=h_\theta(x)$,那么$P(y=0|x;\theta)=1-h_\theta(x)$

因为y不是0就是1，我们可以将这两个式子压缩成(这是一个trick，可以方便后面计算，分别将0，1代入发现原两式成立)
$$
P(y|x;\theta)=h_\theta(x)^y\cdot(1-h_\theta(x))^{(1-y)}
$$
同样的，使用极大似然估计

$$
L(\theta)=P(Y|X;\theta)

=\prod_{i=1}^mP(y^{(i)}|x^{(i)};\theta)

=\prod_{i=1}^mh_\theta(x)^y\cdot(1-h_\theta(x))^{(1-y)}
$$
取对数
$$
l(\theta) = logL(\theta)

=log\prod_{i=1}^mh_\theta(x)^y\cdot(1-h_\theta(x))^{(1-y)}

=\sum_{i=1}^my^{(i)}\cdot log(h_\theta(x^{(i)}))+(1-y^{(i)})\cdot log(1-h_\theta(x^{(i)}))
$$
现在我们需要选择$\theta$使$l(\theta)$最大，我们可以使用p2中的batch gradient descent
$$
\theta_j:=\theta_j+\alpha \cdot\frac{\partial l(\theta)}{\partial\theta_j}
$$
不同于线性回归中的梯度下降法公式的减法，这里的梯度下降用加法，因为线性回归要使损失函数$J(\theta)$最小化，而这里要使极大似然函数$l(\theta)$最小化。同样的可以用开口向上（线性回归）的二次函数 与开口向下（现在）的二次函数图像来直观理解。

另外一个选择sigmoid函数(logistic函数)的原因是，它的似然函数是一个凸函数，因此它也没有局部最大值，只有全局最大值。

经过计算（详见notes：http://cs229.stanford.edu/notes/cs229-notes1.pdf  p18）

$$
\frac{\partial l(\theta)}{\partial\theta_j}=(y-h_\theta (x))x_j
$$
因此

$$
\theta_j:=\theta_j+\alpha \cdot (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}
$$

仔细观察发现这与线性回归中梯度下降法的迭代公式一样。
这是这种模型的普遍性质，这种模型被称为generalized linear model

注意，逻辑回归不同于线性回归，它没有normal equation。

### 牛顿法 Newton's Method
梯度下降法通常每次更新迭代都是一小步一小步，牛顿法更新步子更大，但每次更新需要更多计算。

牛顿法解决的问题是，对于一个函数f，怎么找到参数$\theta$,使得$f(\theta)=0$

对于逻辑回归，我们要使似然函数$l(\theta)$最大化，就要使$l(\theta)$的一阶导数$\dot{l(\theta)}=0$，因此可以用到牛顿法。

**牛顿法是基于当前位置的切线来确定下一次的位置**，直观理解见图：https://blog.csdn.net/a493823882/article/details/81416213

迭代公式为
$$
\theta^{(t+1)}:=\theta^{(t)}-\frac{f(\theta^{(t)})}{\dot{f(\theta^{(t)})}}
$$
当应用于逻辑回归，$f(\theta)=\dot{l(\theta)}$时
$$
\theta^{(t+1)}:=\theta^{(t)}-\frac{\dot{l(\theta^{(t)})}}{\ddot{f(\theta^{(t)})}}
$$

牛顿法具有平方收敛性质(quadratic convergence),在零点附近收敛更快。所以迭代次数更少.

以上为$\theta$是数值的情况，将它延伸到一般的向量$\theta \in \mathbb{R}^{n+1}$时，

$$
\theta^{(t+1)}:=\theta^{(t)}+H^{-1}\Delta_\theta l
$$
其中H是Hessian Matrix，(n+1)x(n+1)维矩阵
$$
H_{ij}=\frac{\delta^2 l}{\delta \theta_i \delta \theta_j}
$$
对于高维矩阵，要求Hessian矩阵的逆开销很大，因此对数据特征量较大的问题不推荐使用牛顿法


## p5

perceptron、exponential family、 generalized linear models、 softmax regression(multiclass classfication)

### Perceptron
Perceptron同样是解决二元分类问题

逻辑回归的hypothesis中g(z)用到的是sigmoid函数，而perceptron用的是
$$
g(z)=
\begin{cases}
0&  z\geq 0\\
1& z<0
\end{cases}

h_\theta(x)=g(\theta^Tx)
$$
它的迭代更新公式同样为
$$
\theta_j:=\theta_j+\alpha \cdot (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}
$$
因为是二元分类，当预测正确时，迭代更新项为0，预测错误时，向量$\theta$会往新数据方向移动/远离一小步。


它与逻辑回归非常类似，只是改了激活函数（？）而已
### Exponential Families 指数族
指数族分布是一种概率分布族，它与GLM（下一章的generalized linear models）非常相似

它的PDF概率密度函数可以表示为：
$$
p(y;\eta)= b(y)exp(\eta^T T(y)-\alpha(\eta))
$$
其中y表示数据，$\eta$是自然参数natural parameter，T(y)表示充分统计量sufficient statistics，一般情况下认为它的值是y，b(y)表示基本度量base measure，$\alpha(\eta)$表示对数分割函数log partition,可以看作标准化常量

你可以自己定义T(y),b(y),$\alpha(\eta)$，只要他们满足概率密度的定义，积分为1，他们就属于指数族分布。

例如伯努利分布 Bernoulli distribution(binary data)

参数$\phi$表示事件发生的概率，它的概率密度函数为
$$
P(y,\phi)=\phi^y (1-\phi)^{(1-y)}
$$
我们的目标是将它转化为指数族的表示方法

$$
P(y,\phi)=\phi^y (1-\phi)^{(1-y)}

=exp(log(\phi^y (1-\phi)^{(1-y)}))

=exp[log(\frac{\phi}{1-\phi})y+log(1-\phi)]
$$
它满足指数族的一般表示$p(y;\eta)= b(y)exp(\eta^T T(y)-\alpha(\eta))$

其中b(y)=1,  T(y)=y,  $\eta=log(\frac{\phi}{1-\phi}), \alpha(\eta)=-log(1-\phi)=-log(1-\frac{1}{1+e^{-\eta}})=log(1+e^\eta)$

因此伯努利分布是指数族的一种。

同理，当方差固定时，正态分布也是指数族的一种。其中 $b(y)=\frac{1}{\sqrt{2\pi}}exp(-\frac{y^2}{2}),  T(y)=y,\eta=\mu, \alpha(\eta)=\frac{\mu^2}{2}=\frac{\eta^2}{2}$（详见视频33：00）

如果你的数据是xx——对应的指数族分布：

实数——高斯分布

binary——伯努利分布

count（正整数）——泊松分布(poisson)

R+（正实数）——Gamma、指数分布（exponential） 

#### 指数族的数学性质

1. 关于$\eta$的极大似然估计是凹的(concave)，那么将它的NLL(negative log likelihood)作为损失函数，就是凸的(convex)
2. 期望$E(y;\eta)=\frac{\delta}{\delta\eta}\alpha(\eta)$
3. 方差$Var(y;\eta)=\frac{\delta^2}{\delta\eta^2}\alpha(\eta)$

第2，3条的证明是作业

### 广义线性模型 GLM Generalized Linear Model
GLM算是指数族的延伸

前提Assumptions/ Design choices:
1. $y|x;\theta$~指数族($\eta$)
2. $\eta = \theta^Tx \quad\theta\in\mathbb{R}^n,x\in\mathbb{R}^n$
3. 测试时：$h_\theta (x)=E[y|x;\theta]$

**x经过线性模型GLM$y=\theta^Tx$会生成参数$\eta$，这个参数$\eta$决定了指数族分布的其他参数（$b(y),T(y),\alpha(\eta)等等$）,通过计算这个指数组分布的期望$E[y|\eta]=E[y|x;\theta]$，让它成为hypothesis function$h_\theta (x)$**

注意，训练时我们仅仅关注$\theta$，与其他参数无关，其他参数仅仅在测试/验证时用到。训练时，我们将对数极大似然函数最大化，

#### GLM training
对于任何一个GLM，迭代更新公式都是：
$$
\theta_j:=\theta_j+\alpha \cdot (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}
$$

canonical response function：$g(\eta)=E[y;\eta]$

canonical link function：$\eta=g^{-1}(E[y;\eta])$

以逻辑回归为GLM的例子推导一下。

逻辑回归是二元分类问题，符合伯努利分布，伯努利分布的期望为$\phi$，上一章中推导出$\alpha(\eta)=-log(1-\phi)=-log(1-\frac{1}{1+e^{-\eta}})=log(1+e^\eta)$，根据3个前提，可以得出
$$
h_\theta (x)=E[y|x;\theta]=E(y;\eta)=\frac{\delta}{\delta\eta}\alpha(\eta)

=\frac{1}{1+e^{-\eta}}=\frac{1}{1+e^{-\theta^Tx}}
$$
与逻辑回归中的hypothesis函数一致，这是逻辑回归的激活函数使用sigmoid函数（logistics函数）的原因。



所有这些推导的源头都是来自hypothesis，我们假设问题中的数据y是基于什么分布的，我们根据分布对应的期望求出$h_\theta(x)$，再根据迭代公式用梯度下降法算出$\theta$，

图像的直观理解 详见视频64：00

### Softmax Regression
Softmax Regression也可以用GLM广义线性模型推导出，但这里用交叉熵来说明

这里要解决的问题是多元分类问题，假设共有k类，$x\in\mathbb{R^n}$,label $y=[\{0,1\}^k]$例如[0,0,1,0]（one-hot vector），每个类都有各自的参数$\theta_{class}\in\mathbb{R^n}$,所以参数矩阵$\theta\in\mathbb{R^{k\times n}}$

hypothesis:给出新数据点属于每个类的概率分布，而不是单独某一类的概率。我们要将这个概率分布，向one-hot靠近，即要最小化这两种分布的差异，即最小化这两种分布的交叉熵。

交叉熵cross entropy
$$
CrossEntropy(y,\hat y)=-\sum_{y\in\{all\_class\}}p(y)log(\hat{p(\hat y)})
$$
由于真实的数据y的分布是确定的，也就是one-hot，因此

$$
CrossEntropy(y,\hat y)=-\sum_{y\in\{all\_class\}}p(y)log(\hat p(\hat y))

=-log(\hat p(\hat{y_{true}}))

=-log(\frac{exp(\theta_{true}^T x)}{\sum exp(\theta_{class}^T x) })
$$
可以参考其他人的学习笔记：https://blog.csdn.net/xierhacker/article/details/53364408

## p6 概率论复习

https://blog.csdn.net/u012566895/article/details/51220127

## p7 生成学习算法 Generative Learning Algorithm
notes：http://cs229.stanford.edu/notes2019fall/cs229-notes2.pdf

https://www.cnblogs.com/madrabbit/p/6935410.html

Gaussian discriminant analysis model(GDA)高斯判别模型，生成和判别的对比，朴素贝叶斯

判别模型（discriminative model）：直接学习P(y|x) 从特征直接到结果 

生成学习算法（Generative Learning Algorithm）：学习P(x|y) 和 P(y) 跟据类别来学习每个类别的特征和类的先验。
再根据贝叶斯公式求出P(y|x)
$$
p(y=1|x)=\frac{p(x|y=1)p(y=1)}{p(x)}
$$
其中$p(x)=p(x|y=1)p(y=1)+p(x|y=0)p(y=0)$

### GDA (Gaussian discriminant analysis)
GDA满足两个假设：
1. $x\in\mathbb{R^n}$(drop $x_0=1$ condition)
2. p(x|y)~Gaussian

#### 多元高斯分布
当一个随机变量z满足多元高斯分布时，z~$N(\mu,\Sigma)$时，概率密度函数
$$
p(z)=\frac{1}{(2π)^{\frac{n}{2}}|\Sigma|^\frac{1}{2}}exp(-\frac{1}{2}
(x-\mu)^T\Sigma^{-1}(x-\mu))
$$

向量$\mu$是高斯分布的均值，矩阵$\Sigma$是协方差矩阵,$\Sigma=E[(x-\mu)(x-\mu)^T]$

协方差矩阵对角线元素的值（$x_0\sim x_n$的方差）控制图像起伏的程度，其他值（相关性）控制图像起伏的方向。
均值控制图像中心的位置。

#### 高斯判别分析模型
假设y服从伯努利分布，$P(y)=\phi^y\cdot(1-\phi)^{(1-y)}$。$\phi$是y=1的概率。
因为p(x|y)~Gaussian，
$$
p(x|y=0)=\frac{1}{(2π)^{\frac{n}{2}}|\Sigma|^\frac{1}{2}}exp(-\frac{1}{2}
(x-\mu_0)^T\Sigma^{-1}(x-\mu_0))\\

p(x|y=1)=\frac{1}{(2π)^{\frac{n}{2}}|\Sigma|^\frac{1}{2}}exp(-\frac{1}{2}
(x-\mu_1)^T\Sigma^{-1}(x-\mu_1))
$$
这个模型有4个参数$\phi,\mu_0,\mu_1,\Sigma$

当使用训练集$\{(x^{(i)},y^{(i)})\}_{i=1}^m$训练模型时，对这些参数作极大似然估计(将它们的联合似然函数最大化)
$$
L(\phi,\mu_0,\mu_1,\Sigma)=\prod_{i=1}^mp(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)

=\prod_{i=1}^mp(x^{(i)}|y^{(i)};...)p(y^{(i)};...)
$$

对于判别学习算法，我们是将条件似然函数$L(\theta)=P(Y|X;\theta)=\prod_{i=1}^mP(y^{(i)}|x^{(i)};\theta)$最大化，这是判别学习算法和生成学习算法一个很大的区别。

我们使用极大似然估计（MLE）。为了便于计算，要将$logL(\phi,\mu_0,\mu_1,\Sigma)$最大化，令它关于四个参数导数为零，得到每个参数结果如下。具体推导详见：https://blog.csdn.net/z_feng12489/article/details/81086183

$$
\phi = \frac{\sum_{i=1}^m y{(i)}}{m} = \frac{\sum_{i=1}^m I(y^{(i)}=1)}{m} 

\mu_0=\frac{\sum_{i=1}^m I(y^{(i)}=0)x^{(i)}}{\sum_{i=1}^m I(y^{(i)}=0)}

\mu_1=\frac{\sum_{i=1}^m I(y^{(i)}=1)x^{(i)}}{\sum_{i=1}^m I(y^{(i)}=1)}

\Sigma=\frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T

$$
其中I是指示函数，I{true}=1,I{flase}=0

直观理解：$\phi$是伯努利分布中y=1的概率，$\mu_0$是所有标签为y=0的向量均值，$\mu_1$是所有标签为y=1的向量均值

我们得到了模型的四个参数，也就得到了p(y),p(x|y=0),p(x|y=1),就以可根据贝叶斯公式$p(y=1|x)=\frac{p(x|y=1)p(y=1)}{p(x)}$以及$p(y=0|x)=\frac{p(x|y=0)p(y=0)}{p(x)}$比较大小求出高斯判别分析模型的预测结果。

### 生成学习算法和判别学习算法的对比

如果画出p(y=1|x)的图像，它是一个sigmoid函数

从结论上看，可以推出（作业）：
$$
\begin{cases}
x|y=0 \sim N(\mu_0,\Sigma)\\
x|y=1 \sim N(\mu_1,\Sigma) & => p(y=1|x)=\frac{1}{1+e^{-\Theta^TX}}\\
y \sim Bernoulli(\phi)
\end{cases}

$$

可证明，当x∣y=0
 与 x∣y=0  服从泊松分布时，同样可以推出p(y|x)服从sigmoid函数。
 
说明GDA的假设 强于 逻辑回归的假设

而**学习算法的一般规律是：模型的假设越强，而且模型结果大致正确，就说明模型越好**

当数据量较少的时候，模型准确率很大程度上依赖于假设。

逻辑回归没有normal equation，需要不断iteration，而GDA不用iteration，但是需要计算协方差矩阵。

https://blog.csdn.net/dukuku5038/article/details/82698867?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

https://blog.csdn.net/Fishmemory/article/details/51711114

## Naive Bayes 朴素贝叶斯

朴素贝叶斯也是生成模型的一种，我们将以垃圾邮件分类（为文本分类）为例子。

首先需要将文本转换为向量表示，在这里我们定义一个常用的10000词的词典，那么向量$x\in\{0,1\}^{10000}$，$x_i$可以表示为该文本中是否包含第i个词.

如果不加以假设来制约，x的值有$2^{10000}$种可能性，服从多项式分布，有$2^{10000}-1$个参数。
$$
p(x_1,...,x_{10000}|y)=p(x_1|y)\times p(x_2|x_1,y)\times ...\times p(x_{10000}|x_1,x_2,...,x_{9999})
$$


我们假设x|y之间条件独立(conditionally independent)也叫Naive Bayes Assumption 强独立假设,即
$$
p(x_1,...,x_{10000}|y)=p(x_1|y)\times p(x_2|y)\times ... \times p(x_{10000}|y)

=\prod p(x_i|y)
$$
参数表示如下：
$$
\phi_{j|y=1}=p(x_j=1|y=1)

\phi_{j|y=0}=p(x_j=1|y=0)

\phi_y=p(y=1)
$$
即当标签为y=1时，第j个词出现在文本中的概率

那么参数的联合似然函数为：

$$
L(\phi_y,\phi_{j|y})=\prod_{i=1}^m p(x^{(i)},y^{(i)};\phi_y,\phi_{j|y})
$$
经过计算，详见
https://blog.csdn.net/z_feng12489/article/details/81381572?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task，

极大似然估计（MLE）为
$$
\phi_y = \frac{\sum_{i=1}^m I\{y^{(i)}=1\}}{m}

\phi_{j|y=1} = \frac{\sum_{i=1}^m I\{x_j^{(i)}=1,y^{(i)}=1\}}{\sum_{i=1}^m I\{y^{(i)}=1\}}
$$
直观理解为：很好理解。。。


## p8

Naive bayes(Laplace smoothing, Event models), comments on apply ML, SVM intros

note： http://cs229.stanford.edu/notes2019fall/cs229-notes2.pdf

 https://blog.csdn.net/z_feng12489/article/details/81381572?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
### 复习朴素贝叶斯算法
以识别垃圾邮件为例，将邮件文本转换成$x\in\R^n$,其中$x_j=I{word j appears in email}$, 为了建立生成模型，我们要计算p(x|y)和p(y)，根据GDA(高斯判别分析)，前者服从高斯分布，后者服从伯努利分布，通过MLE（极大似然估计）算出参数$\phi_y, \phi_{j|y=0}, \phi_{j|y=1}$ ,再通过贝叶斯公式，算出p(y|x)

有一个必须要注意的问题，当新的数据中出现了训练数据中从未出现过的词汇时，将无法判断类别。因为
$$
p(y=1|x_{new})=\frac{p(x_{new}|y=1)p(y=1)}{p(x_{new}|y=1)p(y=1)+p(x_{new}|y=0)p(y=0)}

=\frac{0 }{0+0}
$$
为了解决这个问题，我们需要拉普拉斯平滑(Laplace smoothing)

## 拉普拉斯平滑

以计算球队胜率为例，当球队之前的比赛一场未胜，我们不应该简单的预测下一场胜率为零，一个合理的解决方法是胜负场各加一。 

拉普拉斯平滑：当$x\in\{1,...,k\}$，预测$p(x=j)=\frac{\sum I\{x^{(i)}=j\}+1}{m+k}$

在经过拉普拉斯平滑之后，朴素贝叶斯的参数估计变成
$$
\phi_{j|y=1} = \frac{\sum_{i=1}^m I\{x_j^{(i)}=1,y^{(i)}=1\}+1}
{\sum_{i=1}^m I\{y^{(i)}=1\}+2}
$$

当x不是连续变量，而是离散变量时，例如，将连续变量放入区间bucket中，同样可以使用朴素贝叶斯。

## Event models 事件模型
这种假设词典共有10000词，向量$x\in\{0,1\}^{10000}$，$x_i$表示为该文本中是否包含第i个词.的模型被称为Multi-variate Bernoulli Event model多元伯努利事件模型，无论邮件大小长短，它都会被表示为一个固定维数的向量，而且无论一个单词出现多少次，它都只会变现为1。

而若将这10000个词编号，邮件中一个词对应一个数字，文本单词数量n就是向量的维数，这种模型被称为multinomial event model，多项式事件模型，也被称为词袋模型（bag-of-word model）

对应的参数$\phi_{k|y=0}=p(x_j=k|y=0)$，表示文本中第j个单词为词k。
$$
\phi_{k|y=0} = \frac{\sum_{i=1}^m I\{y^{(i)}=0\} \sum_{j=1}^n I\{x_j=k\}+1}
{\sum_{i=1}^m I\{y^{(i)}=0\}+10000}
$$

其中，拉普拉斯平滑分子加1很好理解，分母10000为k的候选数量，现假设词典共10000词，因此$x_j=k$共有10000种可能，每一种为了避免概率为0，都+1.

详见note。

那么我们在什么时候使用朴素贝叶斯模型呢？在机器学习中，朴素贝叶斯模型在很多问题中表现并不如一般的回归模型。它的优点是高效，可以通过现有数据直接通过公式计算，不需要梯度下降法迭代计算。所以当你的问题需要快速但不太准确（quick and dirty），可以使用朴素贝叶斯。

当开始一个机器学习项目时，往往先开始用快速但不太准确的模型来确定大致的准确率。以上所介绍的GDA，朴素贝叶斯都不一定有很高的准确率，但是他们可以用少量的代码和计算量，（例如计数，或者计算均值方差等），就可以达到还不错的准确率。

额外理解：实际上，从训练数据中估计概率$p(x|y)$是非常困难的。由于$x$是一个高维向量，通常存在维数灾难的问题，刻画这个分布需要的参数数量也是指数级别的。朴素贝叶斯所谓的朴素，实际上就是引入了额外看起来很trivial(挑剔)的假设。朴素贝叶斯假设，$x$的每一维的分布是独立的，在例子中，表现为不考虑文本的前后关联（出现位置），只考虑出现的频率。

严格的从贝叶斯统计推导出了拉普拉斯平滑：https://zhuanlan.zhihu.com/p/24291822

### SVM support vector machine 支持向量机

note:http://cs229.stanford.edu/notes2019fall/cs229-notes3.pdf p11

对于那些明显非线性的分类问题，我们不能用一般的线性回归来解决，一个解决方法是将原来的低维数据映射到更高维的数据，例如将原来二维的$x_1.x_2$，变成n维的$x_1,x_2,x_1^2,x_2^2,x_1x_2$等，对高维数据再使用逻辑回归通常可以解决这个问题。但是如何选择新的高维数据的特征(features)是一个很困扰的问题。SVM可以帮助我们选择这些特征，再进行线性分类。SVM还有一个优点是，我们不需要像梯度下降法那样调整学习率等参数。

关于SVM会讲到：
1. optimal margin classifier，适用于可以被线性分割的两组分离的数据
2. kernels 核方法，将低维数据映射到高维数据
3. 不可被线性分割的两组分离的数据

#### functional margin
functional margin是判断一个分类器好坏的标准

对于一个二元分类器，这里以逻辑回归举例，逻辑回归的hypothesis是
$$
h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$
当$\theta^Tx>0,h_\theta(x)=1$，反之亦然

所以在理想情况下，我们希望当数据标签$y^{(i)}=1$时，$\theta^Tx$远远大于0，反之亦然。这就是functional margin的思想。

#### geometric margin

假设对于一个二维的可先行分割的数据，判断两条分界线的好坏标准可以是，这条分界线分别距离两类数据的最短距离。

SVM在低维数据空间的表现，就是寻找geometric margin最好的分界线。

---

在SVM中，我们使用以下标识：

labels $y\in\{-1, +1\}$

$h_\theta(x^{(i)})$输出为$\{-1, +1\}$

$g(z)=
\begin{cases}
1&  z\geq 0\\
-1& z<0
\end{cases}\>$

原本的$h_\theta(x)=g(\theta^Tx)$其中$x\in \R^{n+1}$, 表示为$h_{w,b}(x)=g(w^Tx+b)$其中$x\in \R^{n}$

---

在这样的表示下,我们可以定义对于数据点$\{(x^{(i)},y^{(i)})\}$的functional margin

$$
\hat{\gamma}^{(i)}=y^{(i)}(w^Tx^{(i)}+b)
$$
当y=1时，我们希望$w^Tx^{(i)}+b$尽可能大；

当y=-1时，我们希望$w^Tx^{(i)}+b$尽可能小

因此，我们希望最大化$\hat{\gamma}^{(i)}$，当$\hat{\gamma}^{(i)}>0$，说明预测正确

那么整个数据集的functional margin

$$
\hat{\gamma}=min \ \hat{\gamma}^{(i)}
$$
为了正则化，我们限制$||w||=1$ 

---

同样的，我们定义geometric margin，经过计算边界线离数据点的距离

详见note3:http://cs229.stanford.edu/notes2019fall/cs229-notes3.pdf p14

$$
{\gamma}^{(i)}=\frac{y^{(i)}(w^Tx^{(i)}+b)}{||w||}
$$
我们定义geometric margin

$$
{\gamma}=min \ {\gamma}^{(i)}
$$

我们可以发现，functional margin$\hat{\gamma}^{(i)}$ 和geometric margin$\gamma^{(i)}$的关系为
$$
{\gamma}=\frac{\hat{\gamma}}{||w||}
$$

#### optimal margin classifier 最大间距分类器

那么我们的optimal margin classifier就是选择能使geometric margin$\gamma^{(i)}$最大化的参数$w,b$

用数学表示为：
$$
\max_{\gamma, w, b} \ \gamma

s.t.  \ \frac{y^{(i)}(w^Tx^{(i)}+b)}{||w||}\geq \gamma
$$
不过由于这个函数不是凸函数，不便于用随机梯度法计算，因此通过简单计算（令$||w||=\frac{1}{\gamma}$）可以将它转化为以下形式（详见note3 p16）
$$
\min_{w,b} \frac{1}{2}||w||^2

s.t. \ y^{(i)}(w^Tx^{(i)}+b)\geq1
$$
它是一个凸函数

## p9 SVM

Optimization problem, Representer theorem, Kernels, Examples of kernels

note3：http://cs229.stanford.edu/notes2019fall/cs229-notes3.pdf

---

前文我们提到最大间距分类器可以表示为
$$
\min_{w,b} \frac{1}{2}||w||^2

s.t. \ y^{(i)}(w^Tx^{(i)}+b)\geq1
$$
其中的$x^{(i)}$可以是任意维数的

假设参数w可以被表示为x的线性组合$w=\sum_{i=1}^m \alpha_ix^{(i)}$

具体解释需要表示理论（Representer theorem），在note3中有证明，从直观上理解1：广义线性模型GLM下的梯度下降法公式$\theta_j:=\theta_j-\alpha \cdot (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$就是使参数$\theta$成为训练数据x的线性组合。直观理解2：任意两数据点$x^{(1)},x^{(1)}$间的中垂线的法向量可以表示为$x^{(1)}-x^{(2)}$;

由于$y^{(i)}\in\{-1,1\}$,所以参数也可以表示为$w=\sum_{i=1}^m \alpha_iy^{(i)}x^{(i)}$

$\min_{w,b} \frac{1}{2}||w||^2$也可以表示为$\min_{w,b} \frac{1}{2}w^Tw$

那么由$w=\sum_{i=1}^m \alpha_iy^{(i)}x^{(i)}$代入最大间距分类器表示公式，得到
$$
\min_{w,b} \frac{1}{2}(\sum_{i=1}^m \alpha_iy^{(i)}x^{(i)})^T(\sum_{j=1}^m \alpha_jy^{(j)}x^{(j)})

=\min_{w,b} \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T  x^{(j)}

=\min_{w,b} \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)}<x^{(i)}, x^{(j)}>
$$
其中$<x,z>$表示内积，等同于$x^Tz$,第一个等号是由于$\alpha, y$都是标量。

约束条件$y^{(i)}(w^Tx^{(i)}+b)\geq1$ 变成
$$
s.t. \quad y^{(i)}((\sum_{j=1}^m \alpha_jy^{(j)}x^{(j)})^Tx^{(i)}+b)\geq1

s.t. \quad y^{(i)}(\sum_{j=1}^m \alpha_jy^{(j)}<x^{(j)},x^{(i)}>+b)\geq1
$$
由此，optimal margin classifier 最大间距分类器可以表示为

$$
\min_{w,b} \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)}<x^{(i)}, x^{(j)}>

s.t. \quad y^{(i)}(\sum_{j=1}^m \alpha_jy^{(j)}<x^{(j)},x^{(i)}>+b)\geq1
$$
观察式子可以发现，$\alpha, y$使标量，因此关键是计算特征向量的内积$<x^{(i)}, x^{(j)}>$,当向量维数很高时，为了简便计算，我们需要用到核方法Kernel methods

根据凸优化理论，以上优化问题还能被简化为以下形式
$$
max \sum_i \alpha_i -\frac12\sum_i \sum_j y^{(i)} y^{(j)}\alpha_i \alpha_j<x^{(i)},x^{(j)}>

s.t.\quad \alpha_i \geq0

\sum_i y^{(i)} \alpha_i =0
 $$
这种优化问题被称为"Dual optimization problem"对偶优化问题，推导详见note3 p22和note3 第六章 

那么整理一下，我们要做的：
1. 根据以上优化问题，获得数据（特征向量）的线性组合参数：$\alpha_i, b$
2. 通过$\alpha_i, b$，进行预测

$$
h_{w,b}(x)=g(w^Tx+b)

=g((\sum_{i=1}^m \alpha_iy^{(i)}x^{(i)})^Tx+b)

=g(\sum_{i=1}^m \alpha_iy^{(i)}<x^{(i)},x> +b)
$$

### Kernel trick
kernel trick 的一般步骤是：
1. Write algorithm in terms of <x,z>
2. Let there be a mapping from $x:\R^2\mapsto \phi(x):\R^{100000}$
3. Find a way to compute $K(x,z)=\phi(x)^T\phi(z)$
4. Replace <x,z> in algorithm with K(x,z)

也就是说，将原本低维空间难以计算的<x,z>，映射到高维空间后变得便于计算

举例说明核方法：

假设$x= \begin{bmatrix} x_1\\x_2\\x_3 \end{bmatrix}$
$\phi (x)= \begin{bmatrix} x_1x_1\\x_1x_2\\x_1x_3\\...\\x_3x_3 \end{bmatrix}$ 
$\phi (z)= \begin{bmatrix} z_1z_1\\z_1z_2\\z_1z_3\\...\\z_3z_3 \end{bmatrix}$,显然$x\in\R^n, \phi (x) \in \R^{n^2}$,我们需要花$O(n^2)$来计算$\phi(x)$以及$\phi(x)^T\phi(z)$

而我们定义$K(x,z)=\phi(x)^T\phi(z)$
$$
K(x,z)=\phi(x)^T\phi(z)

=\sum_{i=1}^n \sum_{j=1}^n x_ix_jz_iz_j

=\sum_{i=1}^n x_iz_i \sum_{j=1}^n x_jz_j  

=(x^T z)^2
$$
计算$\phi(x)^T\phi(z)$的复杂度从原来的$O(n^2)$变成了$O(n)$

即核函数$K(x,z)=(x^T z)^2$可以简化高维向量$\phi (x)= \begin{bmatrix} x_1x_1\\x_1x_2\\x_1x_3\\...\\x_3x_3 \end{bmatrix}$ 的内积计算

---

举例：

$x= \begin{bmatrix} 1\\2\\3 \end{bmatrix}$
$y= \begin{bmatrix} 4\\5\\6 \end{bmatrix}$
$\phi (x)= \begin{bmatrix} x_1x_1\\x_1x_2\\x_1x_3\\...\\x_3x_3 \end{bmatrix}$ 求$<\phi (x), \phi (y)>$

原本我们需要求出$\phi (x)=[1,2,3,2,4,6,3,6,9]$

$\phi (y)=[16,20,24,20,25,36,24,30,36]$

然后计算$<\phi (x),\phi (y)>=16+40+72+40+100+180+72+180+324=1024$



而我们现在知道了核函数$K(x,z)=(x^T z)^2$可以简化高维向量$\phi (x)= \begin{bmatrix} x_1x_1\\x_1x_2\\x_1x_3\\...\\x_3x_3 \end{bmatrix}$ 的内积计算

计算过程变成：$<\phi (x),\phi (y)> = K(x,y)=(x^T y)^2= (4+10+18)^2=32^2=1024$

---



同理，高维向量$\phi (x)= \begin{bmatrix} x_1x_1\\x_1x_2\\x_1x_3\\...\\x_3x_3\\\sqrt{2c}x_1 \\\sqrt{2c}x_2\\\sqrt{2c}x_3\\c \end{bmatrix}$ 的内积计算可以用核函数$K(x,z)=(x^T z+c)^2$计算

延伸一下，核函数$K(x,z)=(x^T z+c)^d$可以简化高维向量$\phi(x)$has all$\binom{n+d}{d}$ features of monomial(单项式) up to order d.

**SVM就是Optimal Margin Classifier 和 Kernel trick的结合**

直观理解：原本在低维空间中线性不可分的数据点，通过核方法映射到高维空间后，变得线性可分（在高维空间里线性可分是我们的假设之一），再使用最大间距分类器对它们分类。视频：https://www.youtube.com/watch?v=3liCbRZPrZA


### How wo make kernels?
首先一个直觉是：当两个向量x，z很相似时，$K(x,z)=\phi(x)^T\phi(z)$的值会很大；相反，不相似时，值会很小。（参考向量内积，两向量垂直时内积为零）

根据这个直觉，我们可以定义一个核函数
$$
K(x,z)=exp(-\frac{||x-z||^2}{2\sigma^2})
$$
很显然它符合这个直觉，x和z越接近，值越大。

那么它可不可以成为核函数呢？

在上文核函数的推导过程中，有$K(x,z)=\phi(x)^T\phi(z)$这个定义，因此**核函数必须满足$K(x,x)=\phi(x)^T\phi(x)\geq0$恒成立**。

另外还必须满足一个条件，假设$\{x^{(1)},x^{(2)},...,x^{(d)}\}$是d个数据点，那么我们可以定义一个由d个点之间核函数值所组成的kernel matrix $K \in \R^{d\times d}$，其中$K_{ij}=k(x^{(i)},x^{(j)})=\phi(x^{(i)})^T\phi(x^{(j)})$，那么对任意的向量z，它的二次型
$$
z^TKz=\sum_i \sum_j z_i K_{ij} z_j

=\sum_i \sum_j z_i \phi(x^{(i)})^T\phi(x^{(j)}) z_j

=\sum_i \sum_j z_i \sum_k \phi(x^{(i)})_k\phi(x^{(j)})_k z_j

=\sum_k (\sum_i z_i \phi(x^{(i)})_k)^2 \geq0
$$
K的二次型大于等于0，说明核函数矩阵K是半正定的。

我们同样可以证明当矩阵K对任意点集$\{x^{(1)},x^{(2)},...,x^{(d)}\}$是半正定时，k(x,z)是合格的核函数

这就是**Mercer定理：任何半正定的函数都可以作为核函数（当且仅当）**

而$K(x,z)=exp(-\frac{||x-z||^2}{2\sigma^2})$是半正定函数，因此可以作为核函数。它被称为**高斯核函数（Gaussian Kernel function）**，是除了线性核函数（$K(x,z)=x^Tz, \phi(x)=x$，相当于没有升高维度）之外最常用的核函数。它可以将低维空间升到无限维度的空间，即$\phi(x)\in\R^{\infty}$

因此任何学习算法中含有$<x^{(i)},x^{(i)}>$时，都可以使用核函数，我们前面所学的所有判别算法包括线性回归，逻辑回归，广义线性模型，perceptron等等都可以使用核函数

#### L1 Norm Soft Margin SVM

SVM中最大间距分类器可以表示为
$$
\min_{w,b} \frac{1}{2}||w||^2

s.t. \quad y^{(i)}(w^Tx^{(i)}+b)\geq1
$$


在推导SVM时我们设置了一个数据线性可分的前提，SVM会根据最坏的情况来对数据分类，这表示一个异常数据会对结果造成很大的影响。因此我们不希望模型对数据过拟合(overfit)。

于是我们放宽一点条件限制，也就是允许一些点不满足这个公式，所以对每个点$x_i$引入一个松弛变量（大于等于0）$\xi_i$，使得

$$
s.t. \quad y^{(i)}(w^Tx^{(i)}+b) + \xi_i\geq1
$$
而松弛变量$\xi_i$的大小由目标函数来控制，因此Soft Margin SVM的目标函数为
$$
\min_{w,b,\xi_i} \frac{1}{2}||w||^2+c\sum_{i=1}^m \xi_i

s.t. \quad y^{(i)}(w^Tx^{(i)}+b) \geq1 - \xi_i
$$
其中$\xi_i\geq0$,c为控制松弛变量的参数

再同上文一样，根据表示理论（Representer theorem）以及凸优化理论假设参数w可以被表示为x的线性组合$w=\sum_{i=1}^m \alpha_ix^{(i)}$
将L1 Norm Soft Margin SVM表示为
$$
max \sum_i \alpha_i -\frac12\sum_i \sum_j y^{(i)} y^{(j)}\alpha_i \alpha_j<x^{(i)},x^{(j)}>

s.t.\quad 0 \leq \alpha_i \leq c 

\sum_i y^{(i)} \alpha_i =0
 $$
与原本的SVM目标函数相比，只是对线性组合的系数$\alpha_i$加了一个不大于c限制条件，c是控制松弛变量的参数，具体如何调参将在下章介绍。

#### 一些核函数的例子

多项式核：$K(x,z)=(x^Tz)^d$
高斯核：$K(x,z)=exp(-\frac{||x-z||^2}{2\sigma^2})$

在识别手写数字任务(handwritten digit classification)中，带多项式核或者高斯核的SVM表现良好。

如何设计核函数是SVM算法中很关键的一步。

例如在蛋白质序列分类(protein sequence classfier)任务。蛋白质由氨基酸组成，假设共有26种氨基酸（事实上是20种），分别为A到Z，因此一段蛋白质序列可以表示为BSVVNSRTGS...的一串字母，蛋白质序列可长可短，因此很难确定如何使用核函数，任务目标是根据序列分类。

我们可以构造函数$\phi(x)\in \R^{20^4}$,分别记录AAAA到ZZZZ在序列中出现的次数。（？？？？）

下章将会介绍如何选择参数。

对于SVM的更多理解：https://blog.csdn.net/wu740027007/article/details/102874510

## p10



Bias/Variance, Regularization, Train/dev/test splits, Model selection & Cross validation

note4:http://cs229.stanford.edu/notes2019fall/cs229-notes4.pdf

之前几周我们讲了很多机器学习算法，现在开始应用机器学习的建议。

Bias和variance易于理解，但是难于掌握。

### Bias/Variance 偏差和方差

了解欠拟合 underfit,过拟合 overfit.

underfit：high bias; overfit: high variance

在当今GPU计算盛行的年代，我们经常会对数据过拟合，防止过拟合的一种方法是regularization正则化

### 正则化 Regularization

正则化并不复杂，但是经常用于大部分机器学习算法。

对于线性回归，Cost function代价函数是

$$
\min_{\theta} \quad \frac{1}{2}\sum_{i=1}^m ||y^{(i)}-\theta^Tx^{(i)}||^2
$$
在后面添加正则化项，
$$
\min_\theta \quad \frac{1}{2}\sum_{i=1}^m (y^{(i)}-\theta^Tx^{(i)})^2+\frac{\lambda}{2}||\theta||^2
$$
$\lambda$可以控制正则化的程度，$\lambda$越大，越趋向于欠拟合，当$\lambda$趋近于无穷大时，$\theta$趋近于零向量。

在逻辑回归中，正则化后的Cost function代价函数是
$$
\arg \max_\theta \sum_{i=1}^m \log p(y^{(i)}|x^{(i)};\theta)-\lambda||\theta||^2
$$

SVM通过核函数，在很高维度下（甚至在无限维）线性分类，但是却不会出现很严重的过拟合，是因为最大间距分类器的代价函数$\min_{w,b} \frac{1}{2}||w||^2$在一定程度上与正则化项类似（具体证明非常复杂）。

在介绍朴素贝叶斯算法时提到的文本分类问题，当特征向量x具有很高维度（定义10000个词的词典）而训练样本远远少于维度时，如果用逻辑回归来解决这个问题，会使模型出现过拟合，但是如果使用正则化后的逻辑回归，它的表现会好于朴素贝叶斯。

因此对于高纬度少样本的数据，我们应该利用正则化。

注意：这里的$\lambda$是一个标量，即对所有的参数$\theta$都加以同一制约，我们很难为每一个参数$\theta_i$选择$\lambda_i$，所以最好将特征向量先作归一化（normalization）预处理。

#### 从另一个角度理解正则化

就像p4种从正态分布的极大似然估计算出广义线性模型代价函数中的LMS最小平均二乘一样，我们看看正则化的另一种解读。

假设训练集$s=\{x^{(i)},y^{(i)}\}_{i=1}^m$我们想要估计对应于这个数据集的参数$\theta$，根据贝叶斯定理，
$$
 p(\theta|S)=\frac{p(S|\theta) p(\theta)}{p(S)}
 
 \theta = \arg \max_\theta p(S|\theta) \ p(\theta)
$$
根据广义线性模型

$$
 \theta = \arg \max_\theta (\prod_{i=1}^mP(y^{(i)}|x^{(i)};\theta))p(\theta)
$$
假设参数向量$\theta$服从于正态分布$\theta \sim N(0,\tau^2I)$(先验分布)，即
$p(\theta)=\frac{1}{(2π)^{\frac{n}{2}}|\tau^2I|^\frac{1}{2}}exp(-\frac{1}{2}\theta^T(\tau^2I)^{-1}\theta)$那么我们仿照p4对它取对数，求导为零，求得的结果就是带有正则化项的代价函数。

---

统计学由两派，频率学派(Frequentist)和贝叶斯学派(Bayesions)。

频率学派认为认为我们所观察到的某些现象其背后的分布是确定的，是一直不变的。**我们常用的极大似然估计(MLE)来计算$p(S|\theta)$就是频率学派的方法。**

基于频率学派思想的方法，我们可以发现其很容易过拟合，因为他的目标是尽最大努力来重现当前观察到的数据，这也是这种方法最大的问题之所在。因此如何解决过拟合问题是基于频率思想方法必然要考虑的。通常我们会使用正则化和交叉验证来缓解过拟合问题。

贝叶斯学派倾向于认为世界上所有的事情都是不确定的，而这种不确定型更多是由于观察者自身所储备的先验知识所带来的。因此**对于贝叶斯学派，其通常会基于观察到的事件来假设一个先验分布P(y)，然后利用贝叶斯公式求得后验分布，这被称为Maximum A Posteriori (MAP，最大后验估计)**。

而后验分布我们又可以认为是在得到新的知识x后对先验分布的一个修正。对于贝叶斯学派最受诟病的就是先验分布的确定，因为通常来说先验分布的选择是基于数学运算的便利性方面来选择的而不是基于真正的先验知识来选择，而选择的先验分布是否接近于真实分布又很大程度上决定了模型的好坏，因此在这一点上常常会被频率学派的学者所抨击。]

如果先验是uniform distribution，则贝叶斯方法等价于频率方法。

在MAP中使用一个高斯分布的先验等价于在MLE中采用L2的regularizaton !
    
---

对于训练数据集，通常越复杂的模型（参数越多），训练误差越小，但是太简单或太复杂的模型，泛化误差（验证集误差）都会较大，因此我们应该选择合适的模型，不能太简单（欠拟合）正则化项$\lambda$太大，也不能太复杂（过拟合）$\lambda$太小。那么怎么选择合适的$\lambda$呢？

### Train/dev/test sets

####  Hold-out cross validation
假设有10000个数据的数据集，针对这些数据，提出了一些模型.

例如模型1：$h_\theta(x)=\theta_0+\theta_1x$ 训练参数$\theta$

模型2：$h_\theta(x)=\theta_0+\theta_1x+\theta_2x^2$ 训练参数$\theta$

模型3: 带正则化项的模型， 训练参数$\theta$和$\lambda$

模型4：LWLR(locally weighted linear regression)局部加权线性回归模型，训练参数$\theta$和带宽$\tau$

模型5：soft margin SVM,训练参数$\theta$和控制松弛变量的超参数c

等等

我们将数据集分为3个子集，训练集，开发集，测试集
- 我们在训练集上训练参数，获得hypothesis h函数
- 在开发集上计算误差（为了避免过拟合，不能再训练集算误差）选择模型（超参数）
- 在测试集评估模型


开发集和测试集的区别（个人理解）：
- 开发集虽然没有直接训练模型的参数，但是帮助了模型选择，例如选择训练次数（防止在训练集上过拟合，这个是否“过拟合”的标准就是开发集来决定的），选择超参数等等
- 测试集只是用来评价模型，不能参与模型选择

不是为了发表论文时，有时候测试集和开发集可以是同一个。

这整个过程被称为 (Simple) Hold-out cross validation 留出交叉验证法。因此开发集(developing set)也被称为交叉验证集(cross validation set)

通常训练集，开发集，测试集比例为6:2:2,或者7：3,但是当数据集很大时，可以根据想了解的精确度，选择适当数量的开发集和测试集

#### k-fold cross-validation

对于小规模的数据集，例如100个医疗数据，每一个数据都有训练那价值，这时我们可以使用k折交叉验证。

通常k=10，先将数据集分成k份，假设有5个模型
$$
for (model=1,2,...5){
    for (i=1,2,...k){
        Train (fit parameters) on k-1 pieces
        Test on remaining 1 piece
    }
    Average test error
}
choose model
$$
优点是：每次训练都只会舍弃1/k的数据，缺点是：训练开销比较大

#### leave-one-out cross validation
对于更小规模的数据，例如20个数据，我们使用k=20的k折交叉验证，也就是说每次都训练19个其他数据来预测一个数据，重复20次。

### 特征选择 feature selection

在机器学习的实际应用中，特征数量往往较多，其中可能存在不相关的特征，特征之间也可能存在相互依赖，容易导致如下的后果：
- 特征个数越多，分析特征、训练模型所需的时间就越长。
- 特征个数越多，容易引起“维度灾难”，模型也会越复杂，其推广能力会下降。

一种防止过拟合的方法是，减少特征，即使用特征的一个子集作为新的特征向量

对n个特征来说，每个特征都存在选择和不选择的情况，因此特征选择一共存在$2^n$个子集。

有一种特征选择的算法是前向搜索算法（forward search）:

一开始选择的特征向量是零向量，计算加入每一个特征后，对模型性能的提升，将提升最大的特征加入特征向量，不断重复。

## p11 Learning Theory
- setup / assumption
- Bias & Variance
- Approximation error & Estimation error
- Empirical Risk Minimizer
- Uniform convergence
- VC dimension

### Assumption
1. 数据集（包括训练集和测试集）都服从同一分布
2. 样本之间互相独立

### Bias & Variance
从另一个角度理解偏差和方差

根据频率学派，假设我们的模型正确，所有数据点实际上是根据一个真实存在，但是我们不知道的常数参数$\theta^*$加随机扰动项生成的，即我们的现有的m点的数据集也是这个模型生成的m个样本。我们根据现有的m个数据的数据集，经过学习算法，得到预测的参数$\hat \theta$

这是一次实验，将以上过程重复n次（随机生成m个数据，将数据集经过学习算法得到预测参数$\hat \theta$）从这n次的结果，可以看出偏差和方差，如图所示：https://blog.csdn.net/u014675538/article/details/77413684

（自己的理解）假设真实数据生成模型是二次的，$y=2x+3x^2$,那么在三维参数空间里$\theta^*=(2,3,0)$，如果用线性模型的算法（欠拟合）实验n次，得到的n个结果可能在$\theta=(3.9,0,0)\sim(4.1,0,0)$附近小范围均匀分布,因此说这个模型算法是high bias, low variance；如果用三次模型算法（过拟合）实验n次，得到的n个结果可能是（为了便于理解）$\theta=(2,3,-5)\sim(2,3,5)$之间均匀分布，因此说这个模型算法是low bias, high variance.

随着样本容量的增加，参数估计具有：渐近无偏性、渐近有效性、一致性。
1. 无偏性(unbiased)：$E[\hat \theta]=\theta^*$样本统计量的数学期望等于被估计的总体参数的值 。总体参数的实际值与其估计值相等时,估计量具有无偏性。
2. 有效性：对同一总体参数的两个无偏点估计量，有更小标准差的估计量更有效。
3. 一致性(consistency)：当$m\rightarrow \infin,\hat \theta \rightarrow \theta^*$。随着样本容量的增大，估计量的值越来越接近被估计的总体参数 。（同直方图表示的话就是，当m越大，中间的突起越来越高）

### Fighting variance
怎么解决high variance 问题？
- 增加数据量m
- 正则化
- 降低模型复杂度。比如决策树模型中降低树深度、进行剪枝等。

---
### Approximation error & Estiamtion error

在hypothesis space中，假设
- g:Best possible hypothesis
- h*:Best in class$H$(H 表示这个模型所有参数h所组成的集合，例如所有逻辑回归模型的集合)
- $\hat h$: learnt from finite data
- E(h):Risk / Generalization error = $E_{(x,y)\sim D}[I(h(x) \neq y)]$对于某hypothesis的真实误差
- $\hat E_S(h)$:Empirical Risk = $\frac1m\sum_{i=1}^m I(h(x) \neq y)$ 对于某hypothesis的通过数据集S推算得到的误差
- E(g) = Bayes error / Irreducible error 由噪声带来的不可约减的误差
- $E(h^*)-E(g)$ = Approximation error 近似误差：度量与最优误差之间的相近程度
- $E(\hat h)-E(h^*)$ = Estiamtion error 估计误差： 度量预测结果与最优结果的相近程度
- 
注意：$E[\hat E_S(h)]=E(h)$,$\hat E_S(h)$不是一定小于$E(h)$，$\hat E_S(h)$只是根据一个数据集样本推算的。

我们可以推出
$$
E(\hat h)= Estiamtion\ error + Approximation\ error + Irreducible\ error

=(Est\ Var + Est\ Bias) + Approx\ error + Irreducible\ error

=Est\ Var + (Est\ Bias + Approx\ error) + Irreducible\ error

= Var + Bias + Irreducible\ error
$$

Approximation error 近似误差是由模型选择造成的，Estiamtion error 估计误差是由数据量不够造成的， Irreducible error 不可约减的误差是由数据生成时自带的噪声造成的

正则化的作用是缩小假设空间集合H，但是会不会增加bias我们并不知道。

理解机器学习中的偏差与方差: https://blog.csdn.net/simple_the_best/article/details/71167786


#### Fight High Bias
- 增加新特征。比如挖掘组合特征、上下文特征、ID类特征。
- 增加模型复杂度。比如在线性模型中增加高次项，在神经网络中增加网络层数或神经元个数。
- 减少或去除正则化系数。比如L1、L2、dropout等。


理解算法与模型的关系：类似妈妈教孩子识字。 妈妈教孩子认字，那一个个的汉字就是数据，妈妈教孩子的过程就是训练的过程，妈妈用的方法就是算法，孩子最后就成了一个能够认识不同字的模型。
https://blog.csdn.net/sy20173081277/article/details/82526262?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

### ERM Empirical Risk Minimizer 经验风险最小化

ERM是一个算法。我们对数据集$\{(x^{(i)},y^{(i)})\}_{i=1}^m$使用ERM算法，得到$\hat h_{ERM}$
$$
\hat h_{ERM} = \arg \min_{h \in H} \frac1m \sum_{i=1}^m I\{h(x^{(i)})\neq y^{(i)}\}
$$

假设H是逻辑回归模型hypothesis空间，那么通过ERM算法可以近似得到逻辑回归的代价函数。

### uniform convergence
我们关注两个问题：
- 比较$\hat E(h)$和$E(h)$，即训数据练集的好坏来体现出整体数据误差的信息
- 比较$E(\hat h)$和$E(h^*)$

为了解决这两个问题，我们需要用到两个引理
- 联合界，the union bound 假设A1,A2,…,Ak为k个不同的事件（可能独立也可能不独立）.那么$P(A_1\cup...\cup A_k)\leq P(A_1)+...+P(A_k)$
- Hoeffding Inequality（Hoeffding 不等式）
  设 Z1, ... , Zm 是 m 个独立同分布（同伯努利分布）的随机变量，即$Z_1,...,Z_k\sim Bernoulli(\phi)$
$\hat\phi = \frac1m\sum_{i=1}^m Z_i$是这些随机变量的均值，则对margin$\gamma > 0$,$P(|\hat \phi - \phi|>\gamma)\leq 2exp(-2\gamma^2m)$。当数据量m越大，误差超过边界的概率越小。

### VC维
详见：https://blog.csdn.net/m0_37687753/article/details/81116528?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task


