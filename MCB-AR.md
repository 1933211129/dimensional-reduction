# 基于极大相容块的不完备信息处理新方法及其应用

王 敬 前 1 ,张 小 红 1,2*

（1.陕西科技大学数学与数据科学学院，西安，710021；

2.陕西省人工智能联合实验室,陕西科技大学，西安，710021）

摘 要：针对不完备信息提出一种新的基于矩阵方法的极大相容块求取算法与属性约简方法，结合智能分类器给出不完备信息条件下的故障诊断方法. 首先，通过矩阵方法计算不完备决策表中的极大相容块；然后，利用所求得的极大相容块，提出一种新的属性约简算法，并与其他方法做对比；最后，将所提出的基于极大相容块的属性约简方法与智能分类器（支持向量机、随机森林、决策树等）结合，建立优化的智能故障分类器，将它应用于不完备信息条件下的故障诊断. 以汽轮机组的故障诊断为例进行仿真实验，实验结果表明提出的针对不完备信息条件下的故障诊断方法可行、有效.

关键词：极大相容块，覆盖粗糙集，矩阵方法，不完备信息，故障诊断

中图分类号：TP30

文献标志码：A

# A new method of incomplete information processing based onmaximal consistent block and its application

Wang Jingqian1 ,Zhang Xiaohong1,2*

(1. School of Mathematics and Data Science，Shanxi University of Science and Technology，Xi′an，710021，China；2. Shanxi Joint Laboratory of Artificial Intelligence，Shanxi University of Science and Technology，Xi′an，710021，China)

Abstract: In this paper，a new maximum compatible block algorithm and attribute reduction method based on matrix methodare proposed for incomplete information，and the fault diagnosis method under the condition of incomplete information is givenby the maximal consistent block and intelligent classifiers. Firstly，the maximal consistent block in an incomplete decisiontable is calculated by matrices. Then，a new attribute reduction algorithm is proposed based on the maximal consistent blockand compared with other methods. Finally，some optimized intelligent fault classifiers are established by the combinationbetween the proposed attribute reduction method and corresponding classifiers，such as support vector machine，random forestand decision tree. It is applied to fault diagnosis under the condition of incomplete information. Moreover，the fault diagnosisof a steam turbine as an example for the simulation. Experimental results shows that the proposed method is feasible andeffective.

Key words: maximal consistent block, covering ⁃ based rough set, matrix approach, incomplete information, fault diagnosis ofsteam turbine

随着科学技术的进步，现代工业生产向大型化、自动化等方向发展，各设备间的联系也更加紧密. 因此，对设备的故障诊断面临新的挑战，其中

一项是不完备信息条件下的故障诊断. 对于故障诊断，不完备信息［1］ 指诊断系统中某一数据对象或多个数据对象的属性值（一般为条件属性）丢

失、不完全或无法确定，表现为数据信息残缺. 为了保证设备运行的安全稳定，对不完备信息条件下的故障诊断方法进行研究有一定现实意义.

粗糙集理论［2］ 是一种处理不精确、不确定和不完备数据的有效分析理论与方法，已在属性约简［3］ 、特征选择［4］ 、知识发现［5］ 等实际应用方面得到了深入的研究 . 目前，对不完备信息的处理方法主要包括删除法、数据补齐法和模型扩充法等［6-8］ . 删除法和补齐法都是通过对缺失信息进行处理，使不完备系统转换为完备信息系统再加以处理，但这两种方法都有一定局限性 . 扩充模型法则是对经典模型加以拓展，使之适应不完备系统的数据处理，这样既保持了信息系统的完整性，又避免了信息失真. 因此，扩充模型法成为当今处理不完备信息的主流方法.

目前，将经典粗糙集模型进行拓展来处理不完备信息已经吸引了很多学者［9-11］ . 例如，Krysz⁃kiewicz［12］ 提出不完备决策表中的容差关系，将经典粗糙集模型扩充为基于容差关系的粗糙集模型，并基于容差关系提出不完备决策表的属性约简 . Leung and Li［13］ 在 容 差 关 系 的 基 础 上 提 出 极大相容块的概念并将其用于解决不完备信息系统的属性约简问题. 黄治国和王淼［14］ 基于极大相容块概念将原不完备决策表转化为极大相容块最全描述系统，进一步研究了属性约简问题. 然而，在极大相容块最全描述系统时并未考虑原决策表中的决策属性，因此，本文进一步考虑带有决策属性的极大相容块最全描述系统 . 另一方面，极大相容块是不完备决策表的最小知识粒单元，对于不完备决策系统知识约简与决策分析有重要研究意义. 然而，如何设计有效的基本知识粒（极大相容块）获取方法以及在此基础上设计快捷高效的知识约简方法，仍然是当前不完备决策系统数据分析与处理面临的重要问题.

本文通过设计新的不完备决策表中极大相容块的求取方法与属性约简方法，提出基于极大相容块与智能分类器的故障诊断方法，并将其应用于不完备信息条件下的汽轮机组故障诊断. 首先提出不完备决策表中的相关矩阵表示，并用这些矩阵计算极大相容块；然后通过极大相容块将原不完备决策表转化为极大相容块最全描述决策

表，在新的决策表基础上提出基于分辨矩阵的属性约简计算方法；最后，基于所提出的基于极大相容块的属性约简方法构建优化的故障分类器，为解决不完备信息条件下的故障诊断问题提供新方法. 由于汽轮机组是石化、能源、冶金和航空等许多行业中的关键设备，因此，本文针对不完备信息条件下汽轮机组的故障诊断问题进行仿真实验.

# 1 基本概念

本节介绍不完备决策表和极大相容块的相关概念. 令 $U$ 是非空有限集合，称为论域.

定义 $\mathbf { 1 } ^ { [ 1 2 ] }$ 设决策表 $D T =  U , A = M \cup D$ ，$\left. V , f \right.$ ，其 中 ， $U = \left\{ x _ { 1 } , x _ { 2 } , \cdots , x _ { n } \right\}$ 为 论 域 ； $A =$$M \cup D$ 为非空有限属性集， $M$ 和 $D$ 分别为条件属性 和 决 策 属 性 ； $\textstyle V = \bigcup _ { a \in A } V _ { a }$ ， $V _ { a }$ 为 属 性 $a$ 的 值域 ； $f \colon U \times A \to V$ 为 映 射 函 数 ， $f \left( { \boldsymbol { x } } _ { i } , { \boldsymbol { a } } \right) \in V _ { a }$$\left( x _ { i } \in U , a \in A \right)$ . 若 存 在 某 对 象 $x \in U$ 的 属 性$a ( a \in M )$ 上取值缺失（通常缺失值用“*”表示），则称该决策表为不完备决策表.

定 义 $\pmb { 2 } ^ { [ 1 2 ] }$ 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle$ ，属性集 $B \left( B \subseteq M \right)$ 上的容差关系定义为：

$$
\begin{array}{l} T O R (B) = \left\{\langle u, v \rangle \in U \times U: \forall a \in B \rightarrow f (u, a) = \right. \\ f (v, a) \vee f (u, a) = * \vee f (v, a) = * \} \\ \end{array}
$$

容差关系 $T O R ( B )$ 是论域 $U$ 上的一个二元关系，由定义 2可知容差关系 $T O R ( B )$ 满足自反性、对称性，但不满足传递性.

定 义 $\pmb { 3 } ^ { [ 1 2 ] }$ 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle$ ， $\forall x \in U$ 关 于 属 性 集 $B \left( B \subseteq M \right)$上的容差类定义为：

$$
T _ {B} (x) = \left\{y \in U: \langle x, y \rangle \in T O R (B) \right\}
$$

定 义 $\mathbf { 4 } ^ { [ 1 3 ] }$ 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle$ ，B⊆ M，X ⊆ U. 若对 $\forall x , y \in X$均有 $\left. x , y \right. \in T O R \left( B \right)$ 成立，则称 $X$ 为属性集 $B$ 上的相容块；若不存在 $X \subset X ^ { \prime }$ 是 $B$ 上的相容块，则称 $X$ 为 $B$ 上的极大相容块.

把由 $B \subseteq M$ 确定的包含对象 $x \in U$ 的所有极大相容块形成的集合表示为 $C _ { x } ( B )$ . 把由 $B \subseteq M$

确定的所有极大相容块形成的集合表示为 $C \left( B \right)$ .

命 题 $\mathbf { 1 } ^ { [ 1 3 ] }$ 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D$ ，V，f ，B ⊆ M，X ⊆ U. X 为 $B$ 上 的 极大相容块当且仅当 $\begin{array} { r } { X = \bigcap _ { x \in X } T _ { B } \left( x \right) } \end{array}$ .

命题 1表明，极大相容块是块中各元素容差类的交集，它是块中各元素容差类的公共部分.

# 2 基于极大相容块的不完备决策表属性约简

2. 1 极大相容块的矩阵计算方法 极大相容块的计算是利用极大相容块技术在不完备信息系统中进行知识获取的重要前提，但在实际计算中它的求取过程总是复杂且耗时，因此，寻找更简便的计算方法非常必要 . 本节利用矩阵方法，提出新的计算极大相容块的方法. 首先给出不完备决策表中两个矩阵表示的定义.

定义 5 设 $D T = \left. U , A = M \cup D , V , f \right.$ 为不完备决策表，其中 $U = \left\{ x _ { 1 } , x _ { 2 } , \cdots , x _ { n } \right\}$ . 称 $F _ { \mathit { T } ( { \boldsymbol { M } } ) } =$$\left( a _ { i j } \right) _ { n \times n }$ 和 $I F _ { \negmedspace T ( M ) } = \left( b _ { i j } \right) _ { n \times n }$ 分 别 为 关 于 条 件 属 性$M$ 的容差关系矩阵表示和容差类交矩阵表示，其中，

$$
a _ {i j} = \left\{ \begin{array}{l l} 1, (x _ {i}, x _ {j}) \in T O R (M) \\ 0, & \text {否 则} \end{array} \right.
$$

$$
b _ {i j} = \left\{ \begin{array}{l l} T _ {M} \left(x _ {i}\right) \cap T _ {M} \left(x _ {j}\right), & x _ {j} \in T _ {M} x _ {i} \\ \Phi , & \text {否 则} \end{array} \right.
$$

例 1 设不完备决策表 ${ D T } = \left. { U , A = M \cup } { D } \right.$ ，$\left. V , f \right.$ 如表1所示，为了后续的对比研究，选取文献［14-15］中的不完备决策表.

根据定义5可得：

$$
F _ {T (M)} = \left( \begin{array}{c c c c c c c c c c c c} 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{array} \right)
$$


表1 不完备决策表 $D T ^ { [ 1 4 - 1 5 ] }$



Table 1 An incomplete decision table $D T ^ { [ 1 4 - 1 5 ] }$


<table><tr><td>U</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>D</td></tr><tr><td>x1</td><td>*</td><td>0</td><td>0</td><td>*</td><td>*</td><td>0</td><td>2</td><td>0</td><td>1</td></tr><tr><td>x2</td><td>*</td><td>2</td><td>*</td><td>*</td><td>1</td><td>*</td><td>0</td><td>1</td><td>0</td></tr><tr><td>x3</td><td>*</td><td>2</td><td>*</td><td>1</td><td>*</td><td>2</td><td>0</td><td>1</td><td>1</td></tr><tr><td>x4</td><td>*</td><td>2</td><td>*</td><td>1</td><td>1</td><td>2</td><td>0</td><td>1</td><td>1</td></tr><tr><td>x5</td><td>1</td><td>*</td><td>*</td><td>*</td><td>1</td><td>0</td><td>*</td><td>0</td><td>0</td></tr><tr><td>x6</td><td>2</td><td>3</td><td>2</td><td>0</td><td>*</td><td>1</td><td>3</td><td>1</td><td>0</td></tr><tr><td>x7</td><td>2</td><td>3</td><td>2</td><td>0</td><td>1</td><td>*</td><td>3</td><td>1</td><td>1</td></tr><tr><td>x8</td><td>2</td><td>3</td><td>2</td><td>1</td><td>3</td><td>1</td><td>*</td><td>1</td><td>1</td></tr><tr><td>x9</td><td>3</td><td>*</td><td>*</td><td>3</td><td>1</td><td>0</td><td>2</td><td>*</td><td>0</td></tr><tr><td>x10</td><td>3</td><td>2</td><td>1</td><td>*</td><td>*</td><td>0</td><td>2</td><td>3</td><td>0</td></tr><tr><td>x11</td><td>3</td><td>2</td><td>1</td><td>1</td><td>1</td><td>0</td><td>*</td><td>*</td><td>0</td></tr><tr><td>x12</td><td>3</td><td>2</td><td>1</td><td>3</td><td>1</td><td>1</td><td>2</td><td>1</td><td>1</td></tr></table>

关于条件属性 $M$ 的容差类交矩阵表示 $I F _ { T ( M ) }$此处暂不给出，将通过后续提出的矩阵计算方法得到 . 命题 2 与命题 3 给出定义 5 中矩阵 $\boldsymbol { F } _ { \ u { T } ( \boldsymbol { M } ) }$ 和${ \cal { I } } F _ { \ T ( M ) }$ 的性质.

命 题 2 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle$ ，其 中 $U = \left\{ x _ { 1 } , x _ { 2 } , \cdots , x _ { n } \right\}$ . 则$F _ { \mathit { T } \left( M \right) }$ 和 ${ \cal { I } } F _ { \ T ( M ) }$ 均为对称矩阵 .

证 明 令 $F _ { \mathit { T } ( M ) } = \left( a _ { i j } \right) _ { n \times n }$ 和 $I F _ { \neg ( M ) } { = } \left( b _ { i j } \right) _ { n \times n }$因为关于属性集 $M$ 的容差关系 $T O R ( M )$ 满足对称 性 ，所 以 $\left( { { x } _ { i } } , { { x } _ { j } } \right) \in T O R \left( M \right)$ 蕴 含 $\left( { \boldsymbol { x } } _ { j } , { \boldsymbol { x } } _ { i } \right) \in$$T O R ( M )$ . 根据定义5可知，对任意 $1 \leqslant i , j \leqslant n$ 总有 $a _ { i j } = a _ { j i }$ ，故 ${ \cal { F } } _ { T ( M ) }$ 为对称矩阵 . 另一方面，如果$x _ { j } \in T _ { M } \big ( \mathcal { x } _ { i } \big )$ ，则必有 ${ x _ { i } } \in T _ { M } \left( { x _ { j } } \right)$ . 因此，由定义 5 可知， ${ \cal { I } } F _ { \ T ( M ) }$ 为对称矩阵.

命 题 3 设 不 完 备 决 策 表 $D T { = } \langle U$ ，$A = M \cup D , V , f \rangle$ ，并且 $F _ { \negmedspace T ( M ) } = \left( a _ { i j } \right) _ { n \times n }$ . 则 ：

$$
a _ {i j} = \left\{ \begin{array}{l l} 1, x _ {j} \in T _ {M} \left(x _ {i}\right) \\ 0, & \text {否 则} \end{array} \right.
$$

证 明 对 任 意 $x _ { i } , x _ { j } \in U , \left( x _ { i } , x _ { j } \right) \in T O R \left( M \right)$当且仅当 $x _ { j } \in T _ { M } \big ( \mathscr { x } _ { i } \big )$ . 由此可见，该命题成立 .

对 于 任 意 的 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle$ ，为了利用 ${ \cal { F } } _ { T ( M ) }$ 计算 ${ \cal { I } } F _ { \ T ( M ) }$ 提出

如 下 矩 阵 计 算 . 设 布 尔 矩 阵 $A { = } \left( a _ { i j } \right) _ { n \times m } , B { = }$ n× m$\left( b _ { i j } \right) _ { m \times t }$ ，则 $D = A \otimes B = \left( d _ { i j } \right) _ { n \times i }$ ，其中，

$$
d _ {i j} = \left\{1 \leqslant k \leqslant m: a _ {i j} \cdot a _ {i k} \cdot b _ {k j} = 1 \right\}
$$

下面给出由 ${ \cal { F } } _ { T ( M ) }$ 计算 ${ \cal { I } } F _ { \ T ( M ) }$ 的矩阵方法 .

命 题 4 设 不 完 备 决 策 表 $D T { = } \langle U$ ，$A = M \cup D , V , f \rangle$ ，其 中 $U = \left\{ x _ { 1 } , x _ { 2 } , \cdots , x _ { n } \right\}$ . 则$I F _ { \mathit { T } ( M ) } { = } F _ { \mathit { T } ( M ) } \bigotimes F _ { \mathit { T } ( M ) } ^ { \mathit { T } }$ ，其 中 $F _ { T ( M ) } ^ { T }$ 为 $F _ { T ( M ) }$ 的转 置 .

证明 令 $F _ { \mathbf { \Phi } _ { T ( M ) } } = \left( a _ { i j } \right) _ { \mathbf { \Phi } _ { n \times n } } , I F _ { \mathbf { \Phi } _ { T ( M ) } } = \left( b _ { i j } \right) _ { \mathbf { \Phi } _ { n \times n } }$ ，并且 $F _ { \mathit { T } ( M ) } \otimes F _ { \mathit { T } ( M ) } ^ { T } = \left( c _ { i j } \right) _ { { n \times n } }$ . 故 ：

$$
\begin{array}{l} c _ {i j} = \left\{1 \leqslant k \leqslant n: a _ {i j} \cdot a _ {i k} \cdot a _ {j k} = 1 \right\} = \\ \left\{1 \leqslant k \leqslant n: a _ {i j} = a _ {i k} = a _ {j k} = 1 \right\} = \\ \left. \left\{x _ {k} \in U: x _ {j} \in T _ {M} \left(x _ {i}\right) \wedge \left(x _ {k} \in T _ {M} \left(x _ {i}\right) \wedge x _ {k} \in T _ {M} \left(x _ {j}\right)\right) \right\} = \right. \\ \left\{ \begin{array}{c c} T _ {M} (x _ {i}) \bigcap T _ {M} (x _ {j}) & x _ {j} \in T _ {M} (x _ {i}) \\ \Phi & \text {否 则} \end{array} = b _ {i j} \right. \\ \end{array}
$$

因此， $. I F _ { \mathit { T } ( M ) } { = } F _ { \mathit { T } ( M ) } \bigotimes F _ { \mathit { T } ( M ) } ^ { \mathit { T } }$

例 2 继续例 1

$$
I F _ {T (M)} = F _ {T (M)} \otimes F _ {T (M)} ^ {T} =
$$

$$
\left( \begin{array}{c c c c c c c c c c c} \left\{x _ {1} x _ {5} x _ {9} \right\} & \Phi & \Phi & \Phi & \left\{x _ {1} x _ {5} \right\} & \Phi & \Phi & \Phi & \left\{x _ {1} x _ {9} \right\} & \Phi & \Phi & \Phi \\ \Phi & \left\{x _ {2} x _ {3} x _ {4} x _ {1 1} \right\} & \left\{x _ {2} x _ {3} x _ {4} \right\} & \left\{x _ {2} x _ {3} x _ {4} \right\} & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \left\{x _ {2} x _ {1 1} \right\} & \Phi \\ \Phi & \left\{x _ {2} x _ {3} x _ {4} \right\} & \left\{x _ {2} x _ {3} x _ {4} \right\} & \left\{x _ {2} x _ {3} x _ {4} \right\} & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi \\ \Phi & \left\{x _ {2} x _ {3} x _ {4} \right\} & \left\{x _ {2} x _ {3} x _ {4} \right\} & \left\{x _ {2} x _ {3} x _ {4} \right\} & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi \\ \left\{x _ {1} x _ {5} \right\} & \Phi & \Phi & \Phi & \left\{x _ {1} x _ {5} \right\} & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi \\ \Phi & \Phi & \Phi & \Phi & \left\{x _ {6} x _ {7} \right\} & \left\{x _ {6} x _ {7} \right\} & \Phi & \Phi & \Phi & \Phi \\ \Phi & \Phi & \Phi & \Phi & \left\{x _ {6} x _ {7} \right\} & \left\{x _ {6} x _ {7} \right\} & \Phi & \Phi & \Phi & \Phi \\ \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \left\{x _ {8} \right\} & \Phi & \Phi & \Phi \\ \left\{x _ {1} x _ {9} \right\} & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \left\{x _ {1} x _ {9} x _ {1 0} \right\} & \left\{x _ {9} x _ {1 0} \right\} & \Phi \\ \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \left\{x _ {9} x _ {1 0} \right\} & \left\{x _ {9} x _ {1 0} x _ {1 1} \right\} & \left\{x _ {1 0} x _ {1 1} \right\} \\ \Phi & \left\{x _ {2} x _ {1 1} \right\} & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \left\{x _ {1 0} x _ {1 1} \right\} & \left\{x _ {2} x _ {1 0} x _ {1 1} \right\} \\ \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi & \Phi \\
$$

${ \cal { I } } F _ { \ T ( M ) }$ 中元素省略对象间的标点，如 $\left\{ x _ { 1 } x _ { 5 } x _ { 9 } \right\}$表示 $\left\{ x _ { 1 } , x _ { 5 } , x _ { 9 } \right\}$ .

推 论 1 设 不 完 备 决 策 表 $D T { = } \langle U$ ，$A = M \cup D , V , f \rangle .$ 则 $I F _ { \mathit { T } ( M ) } { = } F _ { \mathit { T } ( M ) } \bigotimes F _ { \mathit { T } ( M ) }$ .

证明 由命题 2 可知， $\boldsymbol { F } _ { T ( M ) }$ 为对称矩阵，即$F _ { \mathit { T } ( M ) } = F _ { \mathit { T } ( M ) } ^ { \mathit { T } }$ . 因此，该推论成立 .

在上述矩阵计算的基础上，为了通过矩阵方法计算极大相容块，下面的引理1和引理2、命题5和命题6分别给出极大相容块的等价刻画.

引 理 1 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle , X \subseteq U .$ . 如果 $X \in C _ { x } ( M )$ ，则对任意 $y \in X$ ，总有 $T _ { M } ( x ) \cap T _ { M } ( y ) = X .$ .

证明 由命题1可知，该引理显然成立.

引 理 2 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle , X \subseteq U$ ，其中 $U = \left\{ x _ { 1 } , x _ { 2 } , \cdots , x _ { n } \right\}$ .

则 $X \in C _ { x } ( M )$ 当且仅当以下（1）或（2）成立：

（1）对 $\forall y \in X $ ，总有 $T _ { M } ( x ) \subseteq T _ { M } ( y )$ ；

（2）对 $\forall y \in X - \{ x \}$ 总 有 $T _ { M } ( x ) \cap T _ { M } ( y ) = X .$

证明 由命题 1 可知， $X \in C _ { x } ( M )$ 当且仅当$\begin{array} { r } { X = \bigcap _ { y \in X } T _ { M } \binom { y } { y } } \end{array}$ . 又 由 定 义 3 与 定 义 4 可 知 ，$X \subseteq T _ { M } ( x )$ . 因此， $X \in C _ { x } ( M )$ 当且仅当下列条件之一成立：

（1）当 $X = T _ { M } ( x )$ 时 ，对 $\forall y \in X$ ，总 有$T _ { M } ( x ) \subseteq T _ { M } ( y ) ;$ ；

（2）当 $X \subset T _ { M } ( x )$ 时，对 $\forall y \in X - \left\{ x \right\}$ ，总有$T _ { M } ( x ) \cap T _ { M } ( y ) = X .$

命 题 5 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle$ ， $X \subseteq U$ . 则 $X \in C _ { x } ( M )$ 当且仅当以下（1）或（2）成立：

（1）

（2）

$$
| X | - 1
$$

证明 由引理 2 可知，当 $X = T _ { M } ( x )$ 时，对$\forall y \in X$ 都满足 $T _ { M } ( x ) \subseteq T _ { M } ( y )$ ，即：

$$
\begin{array}{l} \left| \left\{y \in X: T _ {M} (x) \subseteq T _ {M} (y) \right\} \right| = \\ \left\{y \in X: T _ {M} (x) \subseteq T _ {M} (y) = T _ {M} (x) \right\} = \\ \left| T _ {M} (x) \right| = | X | \\ \end{array}
$$

当 $X \subset T _ { M } ( x )$ 时 ， $\forall y \in X - \left\{ x \right\}$ 都 满 足$T _ { M } ( x ) \cap T _ { M } ( y ) = X .$ ，即：

$$
\left| \left\{y \in X; T _ {M} (x) \cap T _ {M} (y) = X \right\} \right| = | X | - 1
$$

故该命题成立 .

命 题 6 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle$ ，对 $\forall x \in U$ ，

$$
\begin{array}{l} C _ {x} (M) = \\ \left. \left\{T _ {M} (y) \cap T _ {M} (x): \left| \left\{T _ {M} (y) \cap T _ {M} (x): y \in T _ {M} (x) \right\} \right| = \right. \right. \\ \left| T _ {M} (y) \cap T _ {M} (x) \right| \vee \left| \left\{T _ {M} (y) \cap T _ {M} (x): y \in T _ {M} (x) \right\} \right| = \\ \left| T M (y) \cap T M (x) \right| - 1 \} \\ \end{array}
$$

证明 由命题 5，

$$
C _ {x} (M) \subseteq \left\{T _ {M} (x) \cap T _ {M} (y): y \in T _ {M} (x) \right\}
$$

再由命题5的（1）和（2）可知，该命题成立.

设集合 $A , B$ 定义如下运算：

$$
\begin{array}{l} \frac {1}{A} \diamond \frac {1}{B} = \frac {1}{A \cap B} \\ \frac {1}{A} \oplus \frac {1}{B} = \left\{ \begin{array}{c c} \frac {1}{A} \oplus \frac {1}{B} & A \neq B \\ \frac {2}{A} & A = B \end{array} \right. \\ \end{array}
$$

特别地，若 $A \neq \varPhi , B = \varPhi$ ，则 ${ \frac { 1 } { A } } \oplus { \frac { 1 } { B } } = { \frac { 1 } { A } }$根据所定义的这两种矩阵运算，下面的定义给出两个矩阵交积的定义.

定义 6 设 $A = \left( a _ { i j } \right) _ { n \times m } , B = \left( b _ { i j } \right) _ { m \times 1 }$ 为元素均为集合的矩阵 . 称 $D = A \nabla B = \left( d _ { i j } \right) _ { m \times 1 }$ 为矩阵$A$ 和 $B$ 的交积，其中 $d _ { i j } = \bigoplus _ { k = 1 } ^ { m } \left( { \frac { 1 } { a _ { i k } } } * { \frac { 1 } { b _ { k j } } } \right) .$( a ik ) b kj

基于上述结果，下面的命题给出极大相容块的矩阵计算方法.

命 题 7 设 不 完 备 决 策 表 $D T = \langle U$ ，$A = M \cup D , V , f \rangle , ~ U = \{ x _ { 1 } , x _ { 2 } , \cdots , x _ { n } \} .$ 若$I F _ { \mathit { T } ( M ) } \nabla U = \left( d _ { i } \right) _ { n \times 1 }$ 且 $d _ { i } = \frac { h _ { 1 } } { d _ { i 1 } } \oplus \cdots \oplus \frac { h _ { t } } { d _ { i t } }$ ，则：

$C _ { x _ { i } } ( M ) = \left\{ d _ { i j } \colon 1 \leqslant j \leqslant t \wedge \left( \left| \ d _ { i j } \right| = h _ { j } \vee \left| \ d _ { i j } \right| = h _ { j } - 1 \right) \right\}$其 中 U = ( U ) n . $U = \left( U \right) _ { n \times 1 }$

证明 由定义5和定义6可知，

$$
\left\{d _ {i j}: 1 \leqslant j \leqslant t \right\} = \left\{T _ {M} \left(x _ {i}\right) \cap T _ {M} (y): y \in T _ {M} \left(x _ {i}\right) \right\}
$$

$h _ { j }$ 表 示 与 $d _ { i j }$ 相 等 的 交 集 $\left( T _ { M } \Big ( x _ { i } \Big ) \cap T _ { M } ( y ) \right)$$y \in T _ { M } \big ( x _ { i } \big ) \Big )$ 个数 . 由命题 6 可知，该命题成立 .

# 例3 继续例1

$$
\begin{array}{l} I F _ {T (M)} \nabla U = \\ \left( \begin{array}{c} \frac {1}{\left\{x _ {1} , x _ {5} , x _ {9} \right\}} \oplus \frac {1}{\left\{x _ {1} , x _ {5} \right\}} \oplus \frac {1}{\left\{x _ {1} , x _ {9} \right\}} \\ \frac {1}{\left\{x _ {2} , x _ {3} , x _ {4} , x _ {1 1} \right\}} \oplus \frac {2}{\left\{x _ {2} , x _ {3} , x _ {4} \right\}} \oplus \frac {1}{\left\{x _ {2} , x _ {1 1} \right\}} \\ \frac {3}{\left\{x _ {2} , x _ {3} , x _ {4} \right\}} \\ \frac {3}{\left\{x _ {2} , x _ {3} , x _ {4} \right\}} \\ \frac {2}{\left\{x _ {1} , x _ {5} \right\}} \\ \frac {2}{\left\{x _ {6} , x _ {7} \right\}} \\ \frac {2}{\left\{x _ {6} , x _ {7} \right\}} \\ \frac {1}{\left\{x _ {8} \right\}} \\ \frac {1}{\left\{x _ {1} , x _ {9} \right\}} \oplus \frac {1}{\left\{x _ {1} , x _ {9} , x _ {1 0} \right\}} \oplus \frac {1}{\left\{x _ {9} , x _ {1 0} \right\}} \\ \frac {1}{\left\{x _ {9} , x _ {1 0} \right\}} \oplus \frac {1}{\left\{x _ {9} , x _ {1 0} , x _ {1 1} \right\}} \oplus \frac {1}{\left\{x _ {1 0} , x _ {1 1} \right\}} \\ \frac {1}{\left\{x _ {2} , x _ {1 1} \right\}} \oplus \frac {1}{\left\{x _ {1 0} , x _ {1 1} \right\}} \oplus \frac {1}{\left\{x _ {2} , x _ {1 0} , x _ {1 1} \right\}} \\ \frac {1}{\left\{x _ {1 2} \right\}} \end{array} \right) \\ \end{array}
$$

然后，根据命题 7可计算出任意对象 $x _ { i } \in U$的所有极大相容块 $C _ { x _ { i } } ( M )$ ，如表2所示.

因此，

表 2 任 意 对 象 $x _ { i } \in U$ 的 所 有 极 大 相 容 块 $C _ { x _ { i } } ( M )$$( i = 1 , 2 , \cdots , 1 2 )$ 

Table 2 $C _ { x _ { i } } ( M ) ( i = 1 , 2 , \cdots , 1 2 )$

<table><tr><td>Cx_i(M)</td><td>i=1,2,...,6</td><td>Cx_i(M)</td><td>i=7,8,...,12</td></tr><tr><td>Cx_1(M)</td><td>{x_1,x_5},{x_1,x_9}</td><td>Cx_7(M)</td><td>{x_6,x_7}</td></tr><tr><td>Cx_2(M)</td><td>{x_2,x_11},{x_2,x_3,x_4}</td><td>Cx_8(M)</td><td>{x_8}</td></tr><tr><td>Cx_3(M)</td><td>{x_2,x_3,x_4}</td><td>Cx_9(M)</td><td>{x_1,x_9},{x_9,x_10}</td></tr><tr><td>Cx_4(M)</td><td>{x_2,x_3,x_4}</td><td>Cx_10(M)</td><td>{x_9,x_10},{x_10,x_11}</td></tr><tr><td>Cx_5(M)</td><td>{x_1,x_5}</td><td>Cx_11(M)</td><td>{x_2,x_11},{x_10,x_11}</td></tr><tr><td>Cx_6(M)</td><td>{x_6,x_7}</td><td>Cx_12(M)</td><td>{x_12}</td></tr></table>

$$
C (M) = \left\{\left\{x _ {1}, x _ {5} \right\}, \left\{x _ {1}, x _ {9} \right\}, \left\{x _ {2}, x _ {3}, x _ {4} \right\}, \left\{x _ {2}, x _ {1 1} \right\}, \right.
$$

$$
\left\{x _ {6}, x _ {7} \right\}, \left\{x _ {8} \right\}, \left\{x _ {9}, x _ {1 0} \right\}, \left\{x _ {1 0}, x _ {1 1} \right\}, \left\{x _ {1 2} \right\} \bigg \}
$$

根据上述的极大相容块计算过程，下面给出不完备决策表中极大相容块的矩阵计算方法，如算法1所示.


算法1 不完备决策表中极大相容块的矩阵生成算法


输入：不完备决策表 $\overline { { D T } } = \big < U , A = M \cup D , V , f \big > .$

输出：论域 $U$ 关于 $M$ 的全体极大相容块 $C ( M )$ .

步骤 1 ${ \overline { { C ( M ) = \Phi } } }$

步骤 2 根据定义 5，计算 F ( )； $F _ { T ( M ) }$

步骤 3 根据推论 1，计算 $I F _ { \neg ( M ) } { = } F _ { \neg ( M ) } \otimes F _ { \neg ( M ) }$

步骤 4 根据命题 5，计算全体极大相容块 $C ( M )$ .

算法1中，默认不完备决策表 $D T$ 为一致决策表 . 步骤 1的时间复杂度为 $O ( 1 )$ 步骤 2的时间复 杂 度 为 $O ( | M | | U | ^ { 2 } )$ ,步骤 3的时间复杂度为$O \left( \left| \boldsymbol { U } \right| ^ { 3 } \right)$ ,步骤4的时间复杂度在最坏的情况下不超过 $O { \big ( } { \big | } U { \big | } ^ { 3 } { \big ) } .$ . 一般情况下 $| M | \leqslant | U |$ . 因此 ，算法 1 的整体时间复杂度为 $O \big ( \big | \boldsymbol { U } \big | ^ { 3 } \big )$ .

2. 2 基于极大相容块的属性约简方法 利用2.1中计算的极大相容块，本节给出不完备决策表的属性约简新方法. 黄治国和王淼［14］ 提出由原不完备决策表转化为极大相容块最全描述系统的方法，但在转化过程中没有考虑决策属性，即所求的属性约简与决策属性无关，这不符合决策表的属性约简规则 . 因此，下面给出极大相容块最全描述决策表的定义.

定义 7 设 ${ D T } = \left. { U } , { A } = { M } \cup { D } , { V } , { f } \right. .$ 为不完备决策表，则称 $C D T =  C ( M ) , A = M \cup D$ ，$V ^ { \prime } , f ^ { \prime } \rangle$ 为 $D T$ 的极大相容块最全描述决策表 . 其中 ， $C \left( M \right) = \left\{ X _ { 1 } , X _ { 2 } , \cdots , X _ { t } \right\}$ 为 关 于 属 性 $M$ 的 极大相容块； $M$ 和 $D$ 分别为条件属性和决策属性；$\textstyle V ^ { \prime } = \bigcup _ { a \in A ^ { \prime } } V _ { a } ^ { \prime }$ ， $V _ { a } ^ { \prime }$ 为 属 性 $a$ 的 值 域 ； $f ^ { \prime } \colon C \left( M \right) \times$$A \to V ^ { \prime }$ 为 映 射 函 数 ，其 中 对 任 意 $a \in A$ ， $Y _ { i } \in$$C \left( M \right) \left( 1 \leqslant i \leqslant t \right) .$ ，

$$
\begin{array}{l} f ^ {\prime} \left(X _ {i}, a\right) = \\ \left\{ \begin{array}{l l} \left\{f (x, a): x \in X _ {i} \right\}, & a \in D \\ f (x, a), & a \in M \wedge \left(\exists x \in X _ {i}, s. t., f (x, a) \neq *\right) \\ *, & a \in M \wedge \left(\forall x \in X _ {i}, f (x, a) = *\right) \end{array} \right. \\ \end{array}
$$

例4 继续例1 例3已计算出不完备信息系统 $D T$ 的所有极大相容块：

$$
\begin{array}{l} C (M) = \left\{\left\{x _ {1}, x _ {5} \right\}, \left\{x _ {1}, x _ {9} \right\}, \left\{x _ {2}, x _ {3}, x _ {4} \right\}, \left\{x _ {2}, x _ {1 1} \right\}, \right. \\ \left\{x _ {6}, x _ {7} \right\}, \left\{x _ {8} \right\}, \left\{x _ {9}, x _ {1 0} \right\}, \left\{x _ {1 0}, x _ {1 1} \right\}, \left\{x _ {1 2} \right\} \bigg \} \\ \end{array}
$$

不妨令：

$$
\begin{array}{l} X _ {1} = \left\{x _ {1}, x _ {5} \right\}, X _ {2} = \left\{x _ {1}, x _ {9} \right\}, X _ {3} = \left\{x _ {2}, x _ {3}, x _ {4} \right\}, \\ X _ {4} = \left\{x _ {2}, x _ {1 1} \right\}, X _ {5} = \left\{x _ {6}, x _ {7} \right\}, X _ {6} = \left\{x _ {8} \right\}, \\ X _ {7} = \left\{x _ {9}, x _ {1 0} \right\}, X _ {8} = \left\{x _ {1 0}, x _ {1 1} \right\}, X _ {9} = \left\{x _ {1 2} \right\} \\ \end{array}
$$

则 $D T$ 的极大相容块最全描述决策表 $C D T =$$\left. C ( M ) , A { = } M \cup D , V ^ { \prime } , f ^ { \prime } \right.$ 如 表 3 所 示 .

定义8 设 $D T = \left. U , A = M \cup D , V , f \right.$ 为不完备决策表， $C D T = \left. C ( M ) , A = M \cup D , V ^ { \prime } , f ^ { \prime } \right.$


表3 DT的极大相容块最全描述决策表 (CDT)



Table 3 The most fully described decision table of DTbased on maximal compatible blocks (CDT)


<table><tr><td></td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td><td>D&#x27;</td></tr><tr><td>X1</td><td>1</td><td>0</td><td>0</td><td>*</td><td>1</td><td>0</td><td>2</td><td>0</td><td>0,1</td></tr><tr><td>X2</td><td>3</td><td>0</td><td>0</td><td>3</td><td>1</td><td>0</td><td>2</td><td>0</td><td>0,1</td></tr><tr><td>X3</td><td>*</td><td>2</td><td>*</td><td>1</td><td>1</td><td>2</td><td>0</td><td>1</td><td>0,1</td></tr><tr><td>X4</td><td>3</td><td>2</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td></tr><tr><td>X5</td><td>2</td><td>3</td><td>2</td><td>0</td><td>1</td><td>1</td><td>3</td><td>1</td><td>0,1</td></tr><tr><td>X6</td><td>2</td><td>3</td><td>2</td><td>1</td><td>3</td><td>1</td><td>*</td><td>1</td><td>1</td></tr><tr><td>X7</td><td>3</td><td>2</td><td>1</td><td>3</td><td>1</td><td>0</td><td>2</td><td>3</td><td>0</td></tr><tr><td>X8</td><td>3</td><td>2</td><td>1</td><td>1</td><td>1</td><td>0</td><td>2</td><td>3</td><td>0</td></tr><tr><td>X9</td><td>3</td><td>2</td><td>1</td><td>3</td><td>1</td><td>1</td><td>2</td><td>1</td><td>1</td></tr></table>

为 $D T$ 的 极 大 相 容 块 最 全 描 述 决 策 表 ，其 中$C ( M ) = \left\{ Y _ { 1 } , Y _ { 2 } , \cdots , Y _ { t } \right\}$ . 则 称 $\boldsymbol { F } _ { C D T } = \left( \boldsymbol { a } _ { i j } \right) _ { t \times t }$ 为极大相容块最全描述决策表的分辨矩阵，其中，

$$
a _ {i j} = \left\{ \begin{array}{l l} \left\{a \in M: f ^ {\prime} \left(Y _ {i}, a\right) \neq f ^ {\prime} \left(Y _ {j}, a\right) \right\}, \\ f ^ {\prime} \left(Y _ {i}, D\right) \neq f ^ {\prime} \left(Y _ {j}, D\right) \\ \Phi , \quad f ^ {\prime} \left(Y _ {i}, D\right) = f ^ {\prime} \left(Y _ {j}, D\right) \end{array} \right.
$$

例5 继续例1

$$
F _ {C D T} = \left( \begin{array}{c c c c c c c c c} \Phi & \Phi & \Phi & \{a b c g h \} & \Phi & \{a b c e f h \} & \{a b c h \} & \{a b c h \} & \{a b c f h \} \\ & \Phi & \Phi & \{b c d g h \} & \Phi & \{a b c d e f h \} & \{b c h \} & \{b c d h \} & \{b c f h \} \\ & & \Phi & \{f \} & \Phi & \{b e f \} & \{d f g h \} & \{f g h \} & \{d f g \} \\ & & & \Phi & \{a b c d f g \} & \{a b c e f \} & \Phi & \Phi & \{d f g \} \\ & & & & \Phi & \{d e \} & \{a b c d f g h \} & \{a b c d f g h \} & \{a b c d g \} \\ & & & & & \Phi & \{a b c d e f h \} & \{a b c e f h \} & \Phi \\ & & & & & & \Phi & \Phi & \{f h \} \\ & & & & & & & \Phi & \{d f h \} \\ & & & & & & & & \Phi \end{array} \right)
$$

运用吸收律生成分辨范式：

$$
\begin{array}{l} (d \vee e) \wedge (f \vee h) \wedge (b \vee c \vee h) \wedge (b \vee e \vee f) \wedge \\ \left(d \vee f \vee g\right) \wedge \left(a \vee b \vee c \vee d \vee g\right) \\ \end{array}
$$

将其转换为：

$$
\begin{array}{l} (b \wedge d \wedge f) \vee (b \wedge e \wedge f) \vee (c \wedge d \wedge f) \vee (b \wedge d \wedge h) \vee \\ \left(c \wedge e \wedge f\right) \vee \left(d \wedge e \wedge h\right) \vee \left(d \wedge f \wedge h\right) \vee \\ (e \wedge g \wedge h) \vee (a \wedge e \wedge f \wedge h) \\ \end{array}
$$

因此，DT的约简集：

$$
\begin{array}{l} R E D = \left\{\left\{b, d, f \right\}, \left\{b, e, f \right\}, \left\{c, d, f \right\}, \left\{b, d, h \right\}, \left\{c, e, f \right\}, \right. \\ \{d, e, h \}, \{d, f, h \}, \{e, g, h \}, \{a, e, f, h \} \} \\ \end{array}
$$

由上述定义，便可求得不完备决策表 $D T$ 的所有约简. 具体算法如算法2所示.

# 算法2 不完备决策表的属性约简算法

输入：原不完备决策表 $D T$ 中论域 $U$ 关于 $M$ 的全体极大相容块 $C \left( M \right)$ .

输出：不完备决策表的全体属性约简集RED.

步骤1 根据定义 7，生成 $D T$ 的极大相容块最全描述决策表 $C D T$ ；

步骤 2 根据定义 8，针对极大相容块最全描述决策表 CDT，生成分辨矩阵 FCDT = (aij) ； $C D T$ $F _ { C D T } = \left( a _ { i j } \right) _ { t \times t }$

步骤 3 生成分辨范式 $D N F ( M ) = \Lambda \left\{ \vee a _ { i j } \colon a _ { i j } \neq \phi \right\}$ ；

步骤 4 将 $D N F ( M )$ 转换为等价的极小吸取范式$D N F ^ { \prime } ( M ) = \left( \Lambda T _ { 1 } \right) \vee \cdots \vee \left( \Lambda T _ { k } \right) ;$ ；

步骤 5 得到约简集 $R E D = \left\{ T _ { 1 } , \cdots , T _ { k } \right\}$ .

步骤 1的时间复杂度最长为 $O \Big ( \big | M \big | \big | C \big ( M \big ) \big | ^ { 2 } \Big )$步 骤 2 的 时 间 复 杂 度 为 $O \Big ( \big | M \big | \big | C \big ( M \big ) \big | ^ { 2 } \Big ) .$ ，步 骤 3的时间复杂度为 $O \Big ( \big | C ( M ) \big | ^ { 2 } \Big )$ ,步骤4和步骤5的时间复杂度都为 $O ( 1 )$ . 因此，算法 2的整体时间复杂度为 $O { \Big ( } { \big | } M { \big | } { \big | } C { \big ( } M { \big ) } { \big | } ^ { 2 } { \Big ) } .$

表4给出本文提出的算法2与其他约简算法的比较，并以例 1 为例比较约简结果 . 可以看出，根据本文提出的算法2，取最简约简为 $\left\{ d , f , h \right\}$ . 若以“约简中属性个数越少越好”为评价指标，则本文算法比其他算法的约简效果都好.


表4 本文算法与其他不完备决策表属性约简算法的比较



Table 4 Comparison of our algorithm with other attri⁃bute reduction algorithms of incomplete decision tables


<table><tr><td></td><td>约简算法1[15]</td><td>约简算法2[14]</td><td>本文算法</td></tr><tr><td>基于容差关系</td><td>是</td><td>否</td><td>否</td></tr><tr><td>基于极大相容块</td><td>否</td><td>是</td><td>是</td></tr><tr><td>基于分辨矩阵</td><td>是</td><td>是</td><td>是</td></tr><tr><td>转换为新的决策表</td><td>否</td><td>是</td><td>是</td></tr><tr><td>考虑决策属性</td><td>是</td><td>否</td><td>是</td></tr><tr><td>例1的约简</td><td>{a,c,d,e,f,h}</td><td>{a,d,f,h}</td><td>{d,f,h}</td></tr></table>

# 3 基于极大相容块与智能分类器的故障诊断方法

结合上节提出的基于极大相容块的属性约简方法与智能分类器，提出不完备信息条件下基于极大相容块与智能分类器的故障诊断方法，并对汽轮机组的故障诊断做仿真实验.

3. 1 不完备信息条件下的故障诊断方法 首先，提出不完备信息条件下基于极大相容块与智能分类器（支持向量机、随机森林、决策树等）的故障诊断方法. 如算法3所示.

算法3 不完备信息条件下的故障诊断

输入：不完备故障诊断信息.

输出：故障诊断结果及决策.

步骤1 不完备初始信息表的形成：从采集的样本数据中提取故障征兆，这些征兆对应的故障类别都是已知的. 将这些征兆作为信息系统的条件属性，对应的故障类别作为信息系统的决策属性，形成不完备故障信息系统IS.

步骤2 样本数据的极大相容块预处理：对信息系统IS进行归一化和离散化处理，构成不完备故障诊断决策表 DT，利用算法 1计算极大相容块 $C \left( M \right)$ ；利用算法 2得到针对极大相容块的最全描述决策表 $C D T$ ，并根据相应的差别矩阵对决策表进行约简，消除冗余的条件属性.

步骤 3 “极大相容块+智能分类器”模型的确定及训练：选定智能分类器模型，建立智能分类器诊断模型 .根据步骤 2中确定的约简构造新的故障数据集并对其进行完备化，然后利用其对诊断模型进行训练.

步骤4 故障诊断：用训练好的智能分类器对测试样本征兆集进行故障分类，得到诊断结果.

图1给出不完备信息条件下基于极大相容块与智能分类器的故障诊断方法的流程图.

![](images/54f0a9c7bc5dbff659df6afb0a298aa2653a742800e7c37de67bd469eaa08011.jpg)



图1 “极大相容块+智能分类器”的故障诊断方法流程



Fig 1 The fault diagnosis method of“Maximal compat⁃ibility block $^ +$ Intelligent classifier”


3. 2 汽轮机组故障诊断仿真实验 为了阐述3.1提出的故障诊断新方法，选择汽轮发电机组常见的振动故障作为诊断实例进行仿真实验. 数据处理及运算所用设备：ThinkPad X390笔记本电 脑 ，CPU 为 Intel 酷 睿 i7 8565U，Windows10 系统. 征兆属性为汽轮发电机组振动信号的频域特征频谱中 $< 0 . 4 f , 0 . 4 { \sim } 0 . 5 f , 1 f , 2 f , \geqslant 3 f ( f$ 为旋转频率）等五个不同频段上的幅值分量能量，分别表示为 $a , b , c , d , e$ ；决策属性 $D$ 表示汽轮机组的故障类别， $D$ 的取值为 1，2和 3，分别对应汽轮机组常见的三种故障：油膜振荡、不平衡、不对中 . 不完备汽轮机故障信息表如表 $5 ^ { [ 1 6 - 1 7 ] }$ ］所示（表中数据已归一化处理）.

# 表5 不完备汽轮机故障信息表


Table 5 Incomplete information of turbine faults


<table><tr><td>U</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>D</td></tr><tr><td>1</td><td>0.052</td><td>0.783</td><td>0.225</td><td>*</td><td>0.013</td><td>1</td></tr><tr><td>2</td><td>0.232</td><td>0.975</td><td>0.314</td><td>0.056</td><td>*</td><td>1</td></tr><tr><td>3</td><td>0.161</td><td>*</td><td>0.285</td><td>0.023</td><td>0.016</td><td>1</td></tr><tr><td>4</td><td>0.106</td><td>0.858</td><td>*</td><td>0.017</td><td>0.028</td><td>1</td></tr><tr><td>5</td><td>*</td><td>0.819</td><td>0.201</td><td>0.016</td><td>0.012</td><td>1</td></tr><tr><td>6</td><td>0.028</td><td>0.061</td><td>0.98</td><td>*</td><td>0.057</td><td>2</td></tr><tr><td>7</td><td>0.045</td><td>0.022</td><td>*</td><td>0.316</td><td>0.065</td><td>2</td></tr><tr><td>8</td><td>0.01</td><td>0.054</td><td>0.875</td><td>0.183</td><td>*</td><td>2</td></tr><tr><td>9</td><td>*</td><td>0.032</td><td>0.923</td><td>0.219</td><td>0.037</td><td>2</td></tr><tr><td>10</td><td>0.023</td><td>*</td><td>0.758</td><td>0.115</td><td>0.019</td><td>2</td></tr><tr><td>11</td><td>0.033</td><td>0.037</td><td>0.386</td><td>0.531</td><td>0.23</td><td>3</td></tr><tr><td>12</td><td>*</td><td>0.023</td><td>*</td><td>0.458</td><td>0.103</td><td>3</td></tr><tr><td>13</td><td>0.012</td><td>*</td><td>0.427</td><td>0.496</td><td>0.175</td><td>3</td></tr><tr><td>14</td><td>0.021</td><td>0.017</td><td>0.298</td><td>0.403</td><td>*</td><td>3</td></tr><tr><td>15</td><td>0.017</td><td>0.056</td><td>0.483</td><td>*</td><td>0.301</td><td>3</td></tr></table>

由于应用粗糙集理论处理决策表时，要求决策表中的值用离散数据表达，因此需要进行连续征兆属性的离散化. 本文根据工程实践以及R语言对数据的分析，采用下述断点来实现表 5中条件属性值的离散化，如表6所示.

根据表6的离散化方法，表5离散化后的不完备 故 障 诊 断 决 策 表 $D T = \left. U , A = M \cup D , V , f \right.$如表7所示.


表6 离散化断点表



Table 6 The discrete breakpoint table


<table><tr><td>M</td><td>0</td><td>1</td><td>2</td></tr><tr><td>a</td><td>[0,0.0197]</td><td>(0.0197,0.0437]</td><td>(0.0437,1]</td></tr><tr><td>b</td><td>[0,0.0353]</td><td>(0.0353,0.302]</td><td>(0.302,1]</td></tr><tr><td>c</td><td>[0,0.309]</td><td>(0.309,0.575]</td><td>(0.575,1]</td></tr><tr><td>d</td><td>[0,0.0953]</td><td>(0.0953,0.345]</td><td>(0.345,1]</td></tr><tr><td>e</td><td>[0,0.025]</td><td>(0.025,0.083]</td><td>(0.083,1]</td></tr></table>


表7 不完备汽轮机故障诊断决策表



Table 7 The incomplete turbine fault diagnosis decision


<table><tr><td>U</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>D</td></tr><tr><td>1</td><td>1</td><td>1</td><td>0</td><td>*</td><td>0</td><td>1</td></tr><tr><td>2</td><td>1</td><td>1</td><td>0</td><td>0</td><td>*</td><td>1</td></tr><tr><td>3</td><td>1</td><td>*</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>4</td><td>1</td><td>1</td><td>*</td><td>0</td><td>0</td><td>1</td></tr><tr><td>5</td><td>*</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>6</td><td>0</td><td>0</td><td>1</td><td>*</td><td>0</td><td>2</td></tr><tr><td>7</td><td>0</td><td>0</td><td>*</td><td>0</td><td>0</td><td>2</td></tr><tr><td>8</td><td>0</td><td>0</td><td>1</td><td>0</td><td>*</td><td>2</td></tr><tr><td>9</td><td>*</td><td>0</td><td>1</td><td>0</td><td>0</td><td>2</td></tr><tr><td>10</td><td>0</td><td>*</td><td>1</td><td>0</td><td>0</td><td>2</td></tr><tr><td>11</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>3</td></tr><tr><td>12</td><td>*</td><td>0</td><td>*</td><td>1</td><td>1</td><td>3</td></tr><tr><td>13</td><td>0</td><td>*</td><td>0</td><td>1</td><td>1</td><td>3</td></tr><tr><td>14</td><td>0</td><td>0</td><td>0</td><td>1</td><td>*</td><td>3</td></tr><tr><td>15</td><td>0</td><td>0</td><td>0</td><td>*</td><td>1</td><td>3</td></tr></table>

根据算法1中的基于矩阵法的极大相容块求取方法，得到关于条件属性集 $M$ 的极大相容块$C \left( M \right)$ ，如表8所示.


表8 关于条件属性集 $M$ 的极大相容块 $C ( M )$



Table 8 The maximal compatibility block $C ( M )$ withrespect to the conditional attribute set M


<table><tr><td>X1</td><td>{1,3,5}</td><td>X6</td><td>{10}</td></tr><tr><td>X2</td><td>{2,4}</td><td>X7</td><td>{11}</td></tr><tr><td>X3</td><td>{6}</td><td>X8</td><td>{12,13}</td></tr><tr><td>X4</td><td>{7,9}</td><td>X9</td><td>{12,14}</td></tr><tr><td>X5</td><td>{8}</td><td>X10</td><td>{13,15}</td></tr></table>

利用算法 2，构造针对极大相容块的最全描述决策表CDT，并通过差别矩阵得到约简集. 此处先给出由差别矩阵得到的极小吸取范式：

$$
\begin{array}{l} (a \wedge c) \vee (b \wedge c) \vee (c \wedge d) \vee (c \wedge e) \vee \\ (a \wedge d \wedge e) \vee (b \wedge d \wedge e) \\ \end{array}
$$

因此， $D T$ 的约简集：

$$
R E D = \left\{\{a, c \}, \{b, c \}, \{c, d \}, \{c, e \}, \{a, d, e \}, \{b, d, e \} \right\}
$$

此处，不妨选 $\{ a , c \}$ 为最简约简 .

根据步骤3，只需将条件属性 $a$ 和 $c$ 的属性值完备化，即可构建新的数据集，该数据集作为训练集进行智能故障分类器训练. 此处智能分类器先选 择 支 持 向 量 机（Support Vector Machine，SVM）. SVM 网格法选择惩罚参数 γ 与核参数 $g$ ，范围均为[-10，10]. 但因为后续对比研究的需要，此处将表5都完备化，完备化的方法采用黄文涛等［16］ 的方法补齐. 完备化后的数据集如表9所示，其中粗体数据表示最终所选的属性值与决策值，加下画线的数据表示补齐的数据.

根据步骤 4，对测试集进行测试，如表 10所示. 用本文提出的“极大相容块 $+$ 智能分类器”的


表9 完备化的数据集



Table 9 The complete datasets


<table><tr><td>U</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>D</td></tr><tr><td>1</td><td>0.052</td><td>0.783</td><td>0.225</td><td>0.028</td><td>0.013</td><td>1</td></tr><tr><td>2</td><td>0.232</td><td>0.975</td><td>0.314</td><td>0.056</td><td>0.017</td><td>1</td></tr><tr><td>3</td><td>0.161</td><td>0.859</td><td>0.285</td><td>0.023</td><td>0.016</td><td>1</td></tr><tr><td>4</td><td>0.106</td><td>0.858</td><td>0.256</td><td>0.017</td><td>0.028</td><td>1</td></tr><tr><td>5</td><td>0.138</td><td>0.819</td><td>0.201</td><td>0.016</td><td>0.012</td><td>1</td></tr><tr><td>6</td><td>0.028</td><td>0.061</td><td>0.98</td><td>0.208</td><td>0.057</td><td>2</td></tr><tr><td>7</td><td>0.045</td><td>0.022</td><td>0.884</td><td>0.316</td><td>0.065</td><td>2</td></tr><tr><td>8</td><td>0.010</td><td>0.054</td><td>0.875</td><td>0.183</td><td>0.045</td><td>2</td></tr><tr><td>9</td><td>0.027</td><td>0.032</td><td>0.923</td><td>0.219</td><td>0.037</td><td>2</td></tr><tr><td>10</td><td>0.023</td><td>0.042</td><td>0.758</td><td>0.115</td><td>0.019</td><td>2</td></tr><tr><td>11</td><td>0.033</td><td>0.037</td><td>0.386</td><td>0.531</td><td>0.23</td><td>3</td></tr><tr><td>12</td><td>0.021</td><td>0.023</td><td>0.399</td><td>0.458</td><td>0.103</td><td>3</td></tr><tr><td>13</td><td>0.012</td><td>0.033</td><td>0.427</td><td>0.496</td><td>0.175</td><td>3</td></tr><tr><td>14</td><td>0.021</td><td>0.017</td><td>0.298</td><td>0.403</td><td>0.202</td><td>3</td></tr><tr><td>15</td><td>0.017</td><td>0.056</td><td>0.483</td><td>0.472</td><td>0.301</td><td>3</td></tr></table>

故障诊断方法只需要对条件属性 $a$ 和 $c$ 以及决策属性 $D$ 构成的测试信息进行测试. 为突出这三组数据，它们在表10中用粗体表示.


表 10 汽轮机组故障的测试集[17-18]



Table 10 A test set of turbine fault diagnosis[17-18]


<table><tr><td>U</td><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>D</td></tr><tr><td>16</td><td>0.161</td><td>0.753</td><td>0.128</td><td>0.006</td><td>0.003</td><td>1</td></tr><tr><td>17</td><td>0.282</td><td>0.905</td><td>0.343</td><td>0.046</td><td>0.028</td><td>1</td></tr><tr><td>18</td><td>0.017</td><td>0.053</td><td>0.75</td><td>0.252</td><td>0.107</td><td>2</td></tr><tr><td>19</td><td>0.045</td><td>0.222</td><td>0.989</td><td>0.386</td><td>0.015</td><td>2</td></tr><tr><td>20</td><td>0.027</td><td>0.127</td><td>0.851</td><td>0.619</td><td>0.252</td><td>2</td></tr><tr><td>21</td><td>0.026</td><td>0.043</td><td>0.357</td><td>0.517</td><td>0.098</td><td>3</td></tr><tr><td>22</td><td>0.023</td><td>0.137</td><td>0.378</td><td>0.421</td><td>0.152</td><td>3</td></tr><tr><td>23</td><td>0.0184</td><td>0.0055</td><td>0.8205</td><td>0.127</td><td>0.0031</td><td>3</td></tr></table>

测试结果如图 2所示 . 图中“o”表示预测的故障结果，“*”表示实际的故障类型. 两者重合部分说明诊断结果正确. 图2a、图2c和图2e分别为采用“极大相容块 $+$ SVM”故障诊断方法时，核函数为不同类型的故障诊断结果. 图2b、图2d和图2f则为直接将数据完备化后只采用SVM的故障

诊断结果，此时的核函数也选用不同的类型.

表 11为“极大相容块+SVM”（见表中黑体字）和 SVM 对测试集的故障分类情况的汇总与对比结果.


表 11 “极大相容块+SVM”和 SVM 对测试集的故障分类情况



Table 11 Fault classification of test datasets by“Maxi⁃mal compatible blocks+SVM”and SVM


<table><tr><td></td><td colspan="2">极大相容块+SVM</td><td colspan="2">SVM</td></tr><tr><td></td><td>准确率</td><td>运行时间(s)</td><td>准确率</td><td>运行时间(s)</td></tr><tr><td>Polynomial</td><td>87.5%</td><td>10.63</td><td>75%</td><td>11.15</td></tr><tr><td>RBF</td><td>87.5%</td><td>14.33</td><td>75%</td><td>15.56</td></tr><tr><td>Sigmoid</td><td>87.5%</td><td>12.64</td><td>75%</td><td>12.89</td></tr></table>

从表 11 可以直观地看出，无论 SVM 故障分类器的核函数选择的是多项式（Polynomial）核函数、径向基（RBF）核函数或感知器（Sigmoid）核函数，未经过极大相容块方法约简的 SVM 故障诊断结果的准确率为 $7 5 \%$ ，而约简后的故障诊断结果的准确率提升到 $8 7 . 5 \%$ ，并且运行时间也有所减少. 这表明经过极大相容块约简的属性对故障

![](images/2d28c6c376958c7cb64883cc9a9b4b26835fc62aafadf9d5bdfafb8af86e2afb.jpg)



(a)极大相容块+SVM (Polynomial)


![](images/ea7f12dc8b73ca49bc1ff5117d97f90be898de8c2628ab493f819d895e95184e.jpg)



(b)SVM (Polynomial)


![](images/61ec33b5c73cb692fb8e314b9f9deacd806e2345e67e621f70bbb6897ace1376.jpg)



(c)极大相容块+SVM(RBF)


![](images/63d7e9d21d9f377baa6b9d1663d7e3f57258608e28e069d03e9d73bdbbe2c57b.jpg)



(b)SVM (RBF)


![](images/4cb7e7ab3eaf25b54106d0b18f1c7b65e47d918bdcb5d0fdbcc4cb375344aafa.jpg)



(e)极大相容块+SVM(Sigmoid)


![](images/d551afbeca982d6e1fca2222f2ec9cb2e0e7667aaa37ad465dfee9f7e3d220fa.jpg)



(f) SVM (RBF)



图 2 “极大相容块+SVM”和 SVM 对测试集的故障分类图



Fig.2 Fault classification diagrams of“Maximal compatibility block +SVM”and SVM


诊断更具有代表和准确性，而且减小了 SVM 的运算量，从而使得 SVM 的诊断效果优于没有约简的诊断结果.

为了进一步说明本文提出的“极大相容块+智能分类器”方法的有效性，选用其他类型的智能分类器进行仿真实验，实验结果如表12所示. 由表可见，“极大相容块 $+$ 随机森林”和“极大相容块 $^ +$ 决策树”（表中黑体字）的准确率都高于不经过属性约简而直接选用相应分类器的准确率. 这些结果充分说明所提出的针对不完备信息条件下的故障诊断方法可行、有效.

# 表 12 其他类型的“极大相容块+智能分类器”的故障分类情况


Table 12 Fault classification by other types of“Maxi⁃mal compatible blocks+Intelligent classififiers”


<table><tr><td>极大相容块+智能分类器</td><td>准确率</td></tr><tr><td>极大相容块+随机森林</td><td>87.5%</td></tr><tr><td>极大相容块+决策树</td><td>87.5%</td></tr><tr><td>随机森林</td><td>75%</td></tr><tr><td>决策树</td><td>62.5%</td></tr></table>

# 4 结 论

本文提出基于矩阵方法的极大相容块的矩阵计算方法，很好地解决了传统计算方法复杂、耗时等问题 . 基于极大相容块，提出不完备决策表的最全描述表，并通过分辨矩阵的方法提出新的约简方法，为不完备信息的处理提供了一种新的处理方法 . 借助提出的约简方法，给出不完备条件下“极大相容块 $+$ 智能分类器”的故障诊断方法，并将其应用在汽轮机组的故障诊断中. 实验结果证明提出的故障诊断方法科学地缩减了智能分类器的输入空间规模，减少了训练的时间，提高了诊断的准确率.

# 参考文献



［1］ 陈泽华，宋波，闫继雄，等 . 基于概念格的不完备信息系统最简规则提取算法 控制与决策， ，(5)：1011-1017. (Chen Z H，Song B，Yan J X，et al.Concise rule extraction algorithm of incomplete infor⁃mation system based on concept lattice. Control andDecision，2019，34(5)：1011-1017.)





［2］ Pawlak Z. Rough sets. International Journal ofComputer and Information Science，1982，11(5)：341-356.





［3］ Chen D G， Dong L J， Mi J S. Incrementalmechanism of attribute reduction based on discerniblerelations for dynamically increasing attribute. SoftComputing，2020，24(1)：321-332.





［4］ Tawhid M A，Ibrahim A M. Feature selection basedon rough set approach，wrapper approach，and binarywhale optimization algorithm. International Journal ofMachine Learning and Cybernetics，2020，11(3)：573-602.





［5］ 韩朝，苗夺谦，任福继，等 . 基于粗糙集知识发现的开放领域中文问答检索. 计算机研究与发展，2018，55(5)：958-967. (Han Z，Miao D Q，Ren F J，et al.Rough set knowledge discovery based open domainChinese question answering retrieval. Journal ofComputer research and Development，2018，55(5)：958-967.)





［6］ 姚晟，汪杰，徐风，等 . 不完备邻域粗糙集的不确定性度量和属性约简 . 计算机应用，2018，38(1)：97-103. (Yao S，Wang J，Xu F，et al. Uncertainty mea⁃surement and attribute reduction in incomplete neigh⁃borhood rough set. Journal of Computer Applica⁃tions，2018，38(1)：97-103.)





［7］ Zheng J，Wang Y M，Zhang K，et al. A dynamicemergency decision ⁃ making method based on groupdecision making with uncertainty information.International Journal of Disaster Risk Science，2020，11(5)：667-679.





［8］ Yang L，Zhang X Y，Xu W H，et al. Multi ⁃granulation rough sets and uncertainty measurementfor multi ⁃ source fuzzy information system.International Journal of Fuzzy Systems，2019，21(6)：1919-1937.





［9］ 孙妍，米据生，冯涛，等 . 变精度极大相容块粗糙集模型及其属性约简 . 计算机科学与探索，2020，14(5)：892-900. (Sun Y，Mi J S，Feng T，et al. Maxi⁃mum consistent block based variable precision roughset model and attribute reduction. Journal of Fron⁃tiers of Computer Science & Technology，2020，14(5)：892-900.)





［10］ Sang B B，Yang L，Chen H M，et al. Generalizedmulti ⁃ granulation double ⁃ quantitative decision ⁃theoretic rough set of multi ⁃ source information





system. International Journal of ApproximateReasoning，2019，115：157-179.





［11］ 杨霁琳，秦克云，裴峥. 不完备决策表中基于相似关系 的 属 性 约 简 . 计 算 机 工 程 ，2010，36(20)：10-12.(Yang J L，Qin K Y，Pei Z. Attributes reductionbased on similarity relation in incomplete decision ta⁃ble. Computer Engineering，2010，36(20)：10-12.)





［12］ Kryszkiewicz M. Rough set approach to incompleteinformation systems. Information Sciences，1998，112(1-4)：39-49.





［13］ Leung Y， Li D Y. Maximal consistent blocktechnique for rule acquisition in incompleteinformation systems. Information Sciences， 2003(153)：85-106.





［ ］ 黄治国，王淼 不完备决策表中基于容差关系的属性约简方法. 微电子学与计算机，2016，33(6)：147-151，156. (Huang Z G，Wang M. Attribute reductionbased on tolerance relation in incomplete decision ta⁃ble. Microelectronics & Computer，2016，33(6)：147-151，156.)





［15］ 鄂旭，邵良杉，周津，等 . 一种新的不完备信息系统属性约简算法 . 重庆邮电大学学报(自然科学版)，2010，22(5)：648-651，664. (E X，Shao L S，Zhou J，





et al. A new attribute reduction algorithm in an in⁃complete information system. Journal of ChongqingUniversity of Posts and Telecommunications (Natu⁃ral Science Edition)，2010，22(5)：648-651，664.）





［16］ 黄文涛，赵学增，王伟杰，等 . 基于不完备数据的汽轮机组故障诊断的粗糙集方法. 汽轮机技术，2004，46(1)：57-59. (Huang W T，Zhao X Z，Wang W J，et al. Rough sets method of fault diagnosis for steamturbine based on incomplete data. Turbine Technolo⁃gy，2004，46(1)：57-59.）





［17］ 李化. 汽轮发电机组振动故障智能诊断模型的理论及方法研究. 博士学位论文. 重庆：重庆大学，1999.(Li H. Research on theory and method of intelligentdiagnosis model for turbine generator vibration fault.Ph.D. Dissertation. Chongqing：Chongqing Universi⁃ty，1999.)





［18］ 邴汉昆 . 基于小波分析和 SVM 的汽轮机非线性振动故障诊断研究. 硕士学位论文. 北京：华北电力大学，2013. (Bing H K. Research on nonlinear vibrationfault diagnosis of steam turbine based on waveletanalysis and SVM. Master Dissertation. Beijing：North China Electric Power University，2013.)



（责任编辑 杨可盛）