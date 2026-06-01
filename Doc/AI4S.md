### 第 7 章 综合实验：AI for Science——自然规律和控制方程的符号回归提取算法研究

#### 7.1 实验目的

1. 掌握符号回归的概念、原理和面临的挑战，了解目前在材料领域常用的符号回归算法 SISSO 的设计思路。
2. 学会对算法中的超参数调优，对算法框架中不同模块的设计根据需求进行调整，分析算法收敛性。
3. 自主构建符号回归算法实现纳米粒子生长动力学中生长级数方程的提取。

#### 7.2 实验背景

**7.2.1 Ai for Science 与符号回归背景介绍**

人工智能（AI）在科学研究中的应用正迅速改变传统的研究范式，为多个领域带来创新突破。在基础科学领域，AI 被广泛用于处理复杂数据、建立预测模型和优化实验设计。例如天文学中分析星系图像，或生物学中预测蛋白质结构（如 AlphaFold）。同时，AI 结合物理模型可模拟复杂系统（如气候变化、材料相变），预测实验结果以减少试错成本。自动化实验平台的出现进一步提升了效率，例如化学合成中的 AI 智能体可自主设计实验方案并优化参数。这些应用表明，AI 不仅能够加速数据处理和模式识别，还能从复杂系统中挖掘人类难以察觉的规律。

符号回归（Symbolic Regression, SR）是机器学习中一种旨在从数据中挖掘数学关系的方法。更具体地说，实验得到了一份 n+1 个量的数据，其每一行具有 $\{x_1, \dots, x_n, y\}$ 的形式，其中 $y = f(x_1, \dots, x_n)$，符号回归的任务是发现未知函数 $f$ 的正确符号表达式，可选择地包含噪声的复杂性。

符号回归作为 AI 中的一种特殊方法，在催化与材料科学中展现出独特价值。当前材料领域研究前沿普遍聚焦于原子尺度微观机制与真实复杂体系，此类研究对象具有显著的多体相互作用特征和量子效应，难以通过传统物理建模方法揭示其内在规律（尤其是现象背后的控制方程）。与常见的“黑箱”神经网络不同，符号回归方法通过设计算法从数据中自动推导出数学表达式，揭示变量间的显式关系，为突破复杂催化过程建模和新型材料设计的理论瓶颈提供了创新工具。

**参考资料**
* https://finance.sina.com.cn/roll/2024-07-06/doc-inccerti3198783.shtml
* https://zh.wikipedia.org/zh-cn/AlphaFold
* https://news.ustc.edu.cn/info/1056/89424.htm
* https://en.wikipedia.org/wiki/Symbolic_regression
* https://www.science.org/doi/10.1126/science.adp6034
* https://www.163.com/dy/article/JHJV8NRE05565IHC.html

**7.2.2 研究问题背景**

在催化领域里，载体负载型纳米催化剂的结构稳定性是决定其工业应用潜力的关键性能指标。此类催化剂通常由活性金属纳米颗粒与多孔载体复合构成，其失活机制主要来源于活性组分的结构演变。研究表明，纳米颗粒在高温反应条件下的生长行为受两种竞争机制调控：奥斯特瓦尔德熟化（Ostwald Ripening, OR）机制和颗粒迁移团聚（Particle Migration and Coalescence, PMC）机制。前者源于小尺寸颗粒表面原子溶解-再沉积的扩散过程，后者则与颗粒在载体表面的二维迁移运动直接相关。

通过物理建模，可以得到控制纳米粒子 OR 熟化过程的微分方程，进而由数值模拟得到纳米粒子的生长曲线（纳米粒子粒径随时间变化关系）。演化曲线经归一化处理后，呈现出独特的规律律特征，由此引出表征生长动力学行为的关键无量纲参数——生长级数 q。可以通过理论推导初步建立该参数与体系温度、表面能等热力学量的函数关系：

$$q = a \frac{\alpha^{1/3} \gamma \Omega}{RkT} + b \Leftrightarrow$$

其中，$\alpha, \gamma, \Omega, R, T$ 分别表示接触角修正项、表面能、原子体积、纳米粒子半径；$a, b$ 为常数。本实验的目标即是搭建一个符号回归算法复现这个公式。

（图：负载在载体上的纳米粒子的两种熟化机理示意图）

**参考资料**
* https://en.wikipedia.org/wiki/Ostwald_ripening
* https://en.wikipedia.org/wiki/Coalescence_(chemistry)
* https://www.science.org/doi/10.1126/science.abi9828

#### 7.3 实验原理

**7.3.1 SISSO 算法简介**

目前前沿的符号回归算法通常多个模块集成、架构复杂，本实验将参考材料领域中常用也是架构相对简单的符号回归算法 SISSO 构建算法基础框架。如图 2 所示，SISSO 核心流程包括特征空间构造、特征筛选和系数拟合三个关键阶段。针对材料科学领域的高维特征挖掘需求，SISSO 通过数学运算扩展特征维度，结合统计筛选与优化策略实现高效方程搜索。

（图：SISSO 算法框架示意图）

首先是特征空间扩展机制，具体来说，若初始输入为 $\alpha, \gamma, \Omega, R, T$ 五个特征，扩充操作作为单变量幂次变换（-1, 1/2, ...）与二元交叉组合，特征空间将按如下顺序拓展：

$$A = \{\alpha, \gamma, \Omega, R, T\},$$
$$B = \{\{a, a^{-1}, a^{\frac{1}{2}}, \dots\}, \{\gamma, \gamma^{-1}, \gamma^{\frac{1}{2}}, \dots\}, \dots\},$$
$$C = \{B, \{\alpha\gamma, \alpha\gamma^{-1}, \alpha\gamma^{\frac{1}{2}}, a^{-1}\gamma, \dots\}, \{\dots\}, \dots\}.$$

从理论层面而言，若能够穷尽输入特征的所有数学组合形式，构建具有最优预测性能的方程将变得相对直接。然而实际应用场景中，面临着双重制约：首要制约源于数学表达式的组合爆炸问题，即便在有限阶数约束下，潜在特征组合的基数仍随特征维度呈现指数级膨胀；其次，直接进行回归分析会产生难以承受的计算复杂度，其时间代价与特征空间维度呈线性增长关系。

SISSO 面对这两个问题的优化策略是：首先通过构造基函数库实现特征空间的系统化拓展，随后运用确定性独立筛选（SIS）技术对高维空间实施预降维，再对筛选过后的特征空间执行精确的符号回归分析。这种分层处理机制在保证模型解释性的同时，显著降低了计算复杂度，使处理超高维特征空间成为可能。

**参考资料**
* https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.2.083802
* https://www.sohu.com/a/339727412_489486

**7.3.2 确定性独立筛选（SIS）**

确定性独立筛选的具体原理是通过计算 Pearson 相关系数

$$\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^n (Y_i - \bar{Y})^2}}$$

来衡量两个特征间的线性相关程度。由于 Pearson 相关系数满足 $p(X) = p(aX + b)$，我们可以进行系数拟合前借助 Pearson 相关系数对庞大的特征空间进行筛选。（思考：方程具有多个维度时 $p(X) \neq p(c_1 X_1 + c_2 X_2 + \dots + b)$，如何设计代码框架利用 Pearson 相关系数对空间进行筛选？）

在代码中，我们将使用矩阵运算一次性计算空间中所有特征的 Pearson 系数。假设特征空间代表的矩阵为 $X$（每一列为一个特征的数据），将其标准化后得到 $X_{std} = \frac{X - X_{mean}}{\sigma(X)}$，其中 $X_{mean}, \sigma(X)$ 为对 $X$ 每列数据进行取平均、方差运算得到的行向量。Python 中矩阵运算的广播机制使得在计算 $\frac{X - X_{mean}}{\sigma(X)}$ 时会自动将 $X_{mean}, \sigma(X)$ 扩充为 $X$ 的大小（对行向量沿列的方向进行复制）。目标特征 $q$ 代表的列向量假设为 $y$，将其标准化后得到 $y_{std} = \frac{y - y_{mean}}{\sigma(y)}$，则 Pearson 相关系数 $p(X) = X_{std}^T y_{std}$。

**参考资料**
* https://blog.csdn.net/m0_51327832/article/details/130122685

**7.3.3 线性回归问题的解析求解**

接下来考虑系数拟合过程，我们关注的方程为线性形式（系数通常被认为是方程未能描述的以及未被揭示的物理信息的综合）：$y = c_1 x_1 + c_2 x_2 + \dots + b$，用矩阵表示即为 $y = Xc + b$，若在 $X$ 矩阵末尾添加全 1 的一列，可以把 $b$ 并入向量 $c$ 中：$y' = X'c'$。在实际处理的回归问题中，使用的数据无不包含噪声（数据中的随机波动或误差，可能来源于测量误差、数据录入错误或环境干扰等）。在本实验中，我们假设噪声服从正态分布 $\epsilon \sim N(0, \sigma^2)$，其概率密度函数为 $p(\epsilon) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{1}{2\sigma^2} \epsilon^2)$。由 $y = X'c' + \epsilon$，给定 $X'$，观测到特定 $y$ 的似然为：

$$P(\mathbf{y}|\mathbf{X}') = \prod_i p(y^{(i)}|\mathbf{x}'^{(i)})$$
$$= \prod_i \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2\sigma^2} (y^{(i)} - \mathbf{c}'^T \mathbf{x}'^{(i)})^2\right)$$
$$= \left(\prod_i \frac{1}{\sqrt{2\pi\sigma^2}}\right) \exp\left(-\frac{1}{2\sigma^2} \sum_i (y^{(i)} - \mathbf{c}'^T \mathbf{x}'^{(i)})^2\right)$$

当 $\sum_i (y^{(i)} - \mathbf{c}'^T \mathbf{x}'^{(i)})^2$，即 $\|y - X'c'\|^2$ 取最小值时，$P(y|X')$ 达到最大。对于给定的 $y, X'$，令 $f(c') = \|y - X'c'\|^2$，该函数有且仅有一个极小值点。令 $f(c')$ 对 $c'$ 的导数为 0（全 0 的列向量），可以解得 $c' = (X'^T X')^{-1} X'^T y$（有兴趣的同学可以尝试推导。提示：$f(c') = \|y - X'c'\|^2 = (y - X'c')^T (y - X'c')$；对向量求导的结果是一个分别对向量的每个分量求导后组成的向量）。

**7.3.4 线性回归问题的神经网络求解**

由于在矩阵过大时对矩阵求导在计算量上是一个难以负担的操作，可以转而使用机器学习的方法来实现：构建一个全连接层，以 $X$ 为输入，以 $y$ 为输出，$c, b$ 作为网络中的权重与参数。将 $f(c')$ 作为损失函数，使用梯度下降法最小化 $f(c')$。实验时会探究这两种系数求解方法在面对不同大小的 $X$ 时的准确性与时间问题。

**表 7.1 超参数调整说明**

| 超参数 | 典型调整范围 | 调整影响 |
| :--- | :--- | :--- |
| 学习率 (lr) | 0.1 ~ 1e-5 | 过大导致优化震荡甚至发散，过小收敛缓慢； |
| batch_size | 32, 64, 128, 256 | 大批次训练快且梯度稳定，但泛化性能可能下降；小批次噪声梯度可能提升泛化能力。 |
| epoch | 10 ~ 100+ | 过少导致欠拟合，过多引发过拟合。 |

#### 7.4 实验环境

**7.4.1 本地计算机上的环境配置指导**

(1) 访问 Anaconda 清华镜像 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/，选择合适的版本（如 Windows 系统推荐最新的 Anaconda3-5.3.1-Windows-x86_64.exe）。建议安装路径不带中文。需要勾选添加到 PATH；
(2) 安装完成后打开命令行（Win+R，输入 cmd，打开命令行）输入：`conda -version` 如果返回版本号（如 conda 23.7.4），说明 Conda 已正确安装并在 PATH 中；
(3) 在命令行中输入 `conda create -n ai4science python=3.9 -y` 创建一个虚拟环境；
(4) 在命令行中输入 `conda activate ai4science` 激活环境，命令行显示：
`(ai4science) C:\Users\yourusername>`
表明激活成功；
(5) 激活成功后分别在命令行中输入命令：
`pip install torch==1.12.0`
`pip install numpy==1.26.4`
`pip install sympy==1.13.1`
`pip install matplotlib`
`pip install pandas`
(6) 安装完外部库后输入命令 `pip list` 可以确认上述外部库已成功安装到虚拟环境中；
(7) 访问 https://rec.ustc.edu.cn/group/58634633/disk 下载 ai4science.tar 压缩包；
(8) 访问 https://code.visualstudio.com/download 下载 VSCode。安装完成后，点击左侧扩展，安装 Python 相关插件，推荐插件包：Python Extension Pack；
(9) 新建一个目录，将解压后的所有脚本文件与数据文件放入其中，在 Vscode 中依次选择 file, open the folder 打开该文件夹；
(10) 按下 Ctrl+Shift+P，输入 Python: Select Interpreter，选择 ai4science 环境；
(11) 选择好环境后打开 main.py 脚本，VSCode 右下角应当显示：
`3.9.21('ai4science':conda)`
(12) 点击 Vscode 右上角的三角形图标运行代码文件（若报错“Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.”，表明 Anaconda 环境与 torch 库存在重复的 libiomp5md.dll 文件。可以选择打开 Anaconda 文件夹搜索 libiomp5md.dll 然后删除该文件；或者使用安装 conda 时选择安装 miniconda）。

**7.4.2 在机房服务器上的任务提交指导**

(1) 登入集群后输入命令 `cp /ghome/gpub/ai4science_code.tar ./` 将代码文件复制到根目录下；（也可使用软件从本地将代码文件复制进根目录）
(2) 使用命令 `mkdir foldername` 创建一个运行代码的文件夹；
(3) 输入命令 `cp ./ai4science_code.tar ./foldername/` 和 `cd foldername` 将代码文件复制进文件夹并进入该文件夹；
(4) 输入命令 `tar -xvf ai4science_code.tar` 解压缩代码文件；
(5) 输入命令 `pwd` 获取当前工作目录的路径，复制命令返回的路径；
(6) 输入命令 `vi main.py` 进入 main.py 脚本的编辑界面，使用方向键（不可使用鼠标滚轮）移动光标至“代码运行目录”模块下（位于代码第 14 行，在键盘上依次按下 "13j" 也可向下移动光标），按下 i 键进入编辑模式，输入 "#" 注释第 14 行，再移动光标至下一行删除注释符号，将 yourfilepath 修改为上一步 pwd 命令返回的路径，日志文件与拟合图像将在该目录下生成。修改好后应如下所示：
```python
...
# 代码运行目录
# path = os.getcwd()
path = "\ghome\chenyilin\test"
...
```
(7) 键盘按下 esc 退出编辑模式，按下 shift + ; 键（相当于 : ），输入 wq（保存并退出）后回车；
(8) 输入命令 `vi test.pbs` 编辑提交任务的脚本文件，脚本文件应如下所示：
```bash
#PBS -N cyltest
#PBS -o yourfilepath/$PBS_JOBID.out
#PBS -e yourfilepath/$PBS_JOBID.err
#PBS -l nodes=1:gpus=1:S
#PBS -r y

cd $PBS_O_WORKDIR
echo Time is `date`
echo Directory is $PWD
echo this job runs on following nodes:
echo -n "Node:"
cat $PBS_NODEFILE
echo -n "Gpus:"
cat $PBS_GPUFILE
echo "CUDA_VISIBLE_DEVICES:"$CUDA_VISIBLE_DEVICES

startdocker -P yourfilepath -c "/usr/bin/python yourfilepath/main.py" etcis:5000/isfishingsnow/ustcai:v2.0
```
(9) 其中 cyltest 是任务的名称，可自行修改，脚本中的 yourfilepath 需要替换成 pwd 命令返回的路径，文件最后需要有一个空行；
(10) 编辑好 main.py 与 test.pbs 后在命令行输入 `qsub test.pbs` 提交任务，输入 `qstat` 命令查看任务运行情况，“Q”表示任务正处在队列中，“R”表示正在运行，“C”表示已完成运行。任务完成后输入 `ls` 命令查看当前路径下的文件，输入 `cat jobid.ghead.err`、`cat log` 命令分别查看任务的 bug 日志和输出结果，若 bug 日志中没有显示报错信息则任务正常运行。
注：本书附录 B.2、B.3 部分附有集群介绍、登录与使用的教程，可参考使用。

#### 7.5 基础代码与数据集准备

**7.5.1 实验提供材料说明**

本实验最初提供类似 SISSO 的一个基础算法框架，但其特征扩展模块采用保守策略，不能有效捕获目标方程 $q = a \frac{\alpha^{1/3} \gamma \Omega}{RkT} + b$ 的数学结构。本实验将从两个方向进行改进：一、首先是严格遵循 SISSO 的特征空间构建原理，重构特征扩展模块的算子组合规则。通过实施单次全空间遍历搜索，重点关注符号空间爆炸问题对计算资源的侵占性；二、引入动态特征空间演化策略，建立“特征生成-筛选-评估-特征生成”的循环，在庞大的搜索空间中进行针对性的探索（思考：许多公式拟合效果很好但物理上完全找不到解释，如何在探索时避开这些无意义的符号空间？）。

**7.5.2 算法基础框架与代码介绍**

本实验准备了一个初始的算法框架及其代码，该框架无法直接得到目标方程，在后续的实验过程中将逐步在此框架的基础上进行测试和改进。

（图：算法初始框架示意图）

代码获取方式：https://rec.ustc.edu.cn/group/58634633/home，存储在群盘中的 ai4science 目录中。数据文件同样位于这个压缩包中，data.csv 是输入的 5 个特征（分别代表 $\Omega, \gamma, R, \alpha, kT$），focus.csv 是输出特征数据。

#### 7.6 实验内容

**7.6.1 搭建环境并熟悉算法框架**

按照教程正确建好 python 环境。激活环境后在 Vscode 中运行 main.py 脚本，查看日志文件与输出图像；
浏览算法全貌，熟悉各个模块内容。

**表 7.2 代码文件说明**

| 代码文件 | 功能描述 |
| :--- | :--- |
| feature_information.py | 定义一个储存特征信息的类 |
| expansion.py | 对初始输入特征进行扩充 |
| sis_ana.py | 使用确定独立筛选方法进行筛选 |
| coefficients_fitting.py | 拟合系数（包含解析与神经网络两种方法） |
| result_sorting.py | 计算特征的拟合效果并进行排序、储存 |
| result_displaying.py | 将结果输出为日志文件 |
| main.py | 主函数 |
| result_plotting.py | 绘制图像 |
| timer.py | 定义一个记录运行时间的类 |

**依赖库参考文档**
* https://pytorch.org/docs/stable/index.html
* https://numpy.org/doc/stable/
* https://docs.sympy.org/latest/index.html
* https://matplotlib.org/stable/tutorials/index
* https://pandas.pydata.org/docs/

**7.6.2 线性回归问题的两种方式求解**

**解析求解**
算法默认使用解析方式进行求解，coefficients_fitting.py 中相关代码如下：
```python
# 进行拟合系数的函数
def fit(x:pd.DataFrame, y:ND):
    ...
    # 若数据转换为tensor格式
    X = torch.tensor(x.to_numpy(), device=device)
    y = torch.tensor(y, device=device)
    ...
    # X的每列数据即为y=a*f(x1,x2,..)+b中的f(x1,x2,..)方程数据
    # 对每列数据分别拟合系数
    for i in range(X.shape[1]):
        # 解析解
        r2, p, loss = analytical_solving(X[:, i], y[:])
```
```python
    # 解析求解函数
    def analytical_solving(X:ND, y:ND, lambda_reg=1e-6):
    ...
```
运行 main.py，重命名日志文件将其保存起来，以作为神经网络解结果的参考。

**神经网络求解**
在 coefficients_fitting.py 的 fit 函数中将解析解对应函数调用注释掉，解除神经网络求解对应注释。
```python
# 进行拟合系数的函数
def fit(x:pd.DataFrame, y:ND):
    ...
    # 若数据转换为tensor格式
    X = torch.tensor(x.to_numpy(), device=device)
    y = torch.tensor(y, device=device)
    ...
    # X的每列数据即为y=a*f(x1,x2,..)+b中的f(x1,x2,..)方程数据
    # 对每列数据分别拟合系数
    for i in range(X.shape[1]):
        # 解析解
        # r2, p, loss = analytical_solving(X[:, i], y[:])
        # 神经网络求解
        r2, p, loss = net_solving(X[:, i], y[:])
    ...
```
运行 main.py，比对解析解与神经网络解的结果。
按如下方式调整 fit 函数中神经网络的超参数使神经网络解的结果收敛到解析解：
```python
def fit(x:pd.DataFrame, y:ND):
    ...
    r2, p, loss = net_solving(X[:, i], y[:], num_epochs = ..., batch_size = ..., lr = ...)
    ...
```
神经网络结果收敛后使用 Timer 类记录两种求解方式的耗时，并分析原因。Timer 类使用示例：
```python
from timer import Timer
...
# 创建Timer类变量并开始计时
ti = Timer()
...需计时代码...
# 停止计时
ti.stop()
# 输出累计用时
print(ti.cumsum())
```
将 sis 模块注释掉，再次调整神经网络至结果收敛并记录分析两种求解方式的耗时。

**7.6.3 重构特征扩展模块**

在提供的基础框架上模仿 expension.py 脚本中的尝试重构特征扩展模块，实施单次全空间遍历搜索（可以添加更多元的组合、指数、对数等运算）。比如若要添加三元组合操作，可仿照 expension.py 脚本中的二元组合部分进行改写：
```python
# 从所有特征中放回抽取三个特征，返回所有可能的组合的索引
combinations = list(itertools.combinations(range(len(out_num)), 3))
# 对于每一种组合
for combination in combinations:
    # 对应组合的三个特征的索引
    idx1, idx2, idx3 = combination
    # 通过combine_3函数进行数据的操作，可模仿代码中已有的combine函数进行编写
    num_c, str_c = combine_3(out_num[idx1], out_num[idx2], out_num[idx3]],
                             out_str[idx1], out_str[idx2], out_str[idx3])
    # 将新产生的特征加入输出中
    out_num.append(num_c)
    out_str.append(str_c)
```
尝试添加各式各样的扩充方式（种类、顺序）以尽可能扩大扩充后的符号空间，记录每种扩充方式添加后空间的膨胀程度（若报错内存不足，理论上定量给出结果即可），最后尝试以此方法学习到目标方程（此处可以不考虑公式的复杂度与可解释性，若能得到 R 方大于 0.95 的公式也算完成）

**7.6.4 构建循环针对性探索符号空间**

搜索前毫无偏好地对输入特征空间进行扩充显然效率低下，因为有效、物理上具备可解释性的特征占扩充后空间的比例过小。很自然的一个改进方向便是先进行初步的扩充，随后由搜索结果指导下一次扩充，使符号空间的探索更具针对性。
按以上思路改进算法需至少设计以下模块：
1. 单次迭代结果如何引导下一次符号空间探索的函数（最简单可设置成直接将输出作为下一次的输入，添加或替换均可，如采用直接添加的方式需注意扩充后的表达式重复问题；也可不更改由输入构造特征空间的模块，而是将输入特征构造出的特征空间与输出之间进行组合，得到依赖于输出的新特征空间）；
2. 储存循环中产生的公式信息的一个变量或者类（若每次输出的特征之间没有关系，可使用一个变量存储；若每次输出之间存在关系，子母关系，可构建一个树状结构类来存储，可参考 https://blog.csdn.net/u013121610/article/details/130514658 实现）；
3. 迭代终止策略（可设置成一定迭代次数后即终止；也可在算法中实现一个滑动窗口，记录算法迭代至今的高分特征，检测循环产生的特征的分数相比窗口是否有提升，若一定迭代次数后仍无提升即终止）；
下面是提供参考的框架设计（非强迫使用）：
```python
...
# 读取数据
data = pd.read_csv('data.csv')
focus = pd.read_csv('focus.csv')

# todo 定义一个判断终止条件的函数
def stop_or_not(...):
    ...
    return True
    ...
    return False

# todo 定义一个整理每次循环结果的函数
def store_result(results_temp, ...):
    ...
    return results

# 迭代次数记录
iter = 1

# 未判断终止时循环继续
continue = True

# 循环
while continue:
    
    # 初始输入的扩充
    if iter == 1:
        data_expanded = expand(data)
    else:
        # todo 思考如何根据上一次的输出构建expand_next函数
        data_expanded = expand_next(data, results_temp)
        
    # 确定性独立筛选
    data_sis = sis(data_expanded, focus.to_numpy(), 10)
    
    # 系数拟合
    r2, coef, loss = fit(data_sis, focus.to_numpy())
    
    # 结果整理
    results_temp = sort_result(data_sis.to_numpy(),
                               data_sis.columns.to_numpy(),
                               r2, coef, loss, 10)
                               
    # todo 构建store_result函数将每次循环的结果储存起来
    results = store_result(results_temp, ...)
    
    # 判断是否终止迭代
    if stop_or_not(...):
        continue = False
    else:
        iter += 1
        
# 输出日志
current_directory = os.getcwd()
display_result(results, 10, current_directory)
...
```
建议先规划整体要设计哪些模块，画出改进后的框架示意图，再编写各模块代码细节。（思考：如何设计模块 2 与扩充方式以增加结果的可解释性，提示：实际中处理的问题存在物理背景，各个输入有其物理含义）
根据以上内容改进算法，尝试得到目标方程。

#### 7.7 实验要求与评分细则

**7.7.1 需要提交的内容**

(1) 修改后的完整代码（含注释）根据 7.6.3 和 7.6.4 实验内容修改后运行算法所需的所有脚本与数据集打包提交即可（两个压缩包，分别对应两个实验内容的代码）。
(2) 实验报告撰写完整的实验报告，包括实验目的、原理、步骤、结果分析、思考题、实验设计建议等模块。结果分析模块中应包含 7.6.2 中两种系数拟合方法结果与效率的比较；7.6.3 中添加的不同处理方式对符号空间膨胀的定量影响，修改后的结果展示与分析；7.6.4 中改进后的算法框架示意图、改进说明、修改后的结果展示与分析等内容。
注：按照实验手册进行实验时若遇到困难或者对符号回归在微观物理化学问题中进一步的应用感兴趣的同学可联系助教（860451061@qq.com）答疑或者了解。

**7.7.2 评分细则**

(1) 代码完整性与注释（10%）
* 代码必须完整，每一部分应有清晰、简洁的注释，确保其他人可以理解代码的功能和实现细节。
(2) 实验结果与分析（65%）
* 两种线性回归问题求解方法结果与效率的比较与分析部分：
    - 能调整神经网络超参数使结果收敛到解析解（5%）
    - 注释 sis 模块前后的结果与效率比较（5%）
* 重构扩充函数部分：
    - 清晰的描述出扩充方式的增加、种类、先后顺序对符号空间膨胀的定量影响（20%）
    - 通过该方法能够发现目标公式或者得到 R 方大于 0.95 的公式（2.5%，不可为了凑答案而特殊化扩充方式）
* 重新设计算法框架，针对性探索符号空间部分：
    - 完成基本模块的设计，确保代码正常运行（15%）
    - 绘制算法框架示意图（5%）
    - 说明各模块（策略）如何设计（10%，设计的合理性）
    - 通过该方法能够发现目标公式（2.5%）
(3) 思考题（15%）
* 在实验报告中完成思考题的回答（思考题没有标准答案，逻辑通顺合理即可）
(4) 报告与文档（10%）
* 提交完整的实验报告，报告格式规范，内容清晰明了。

#### 7.8 思考题

1. 方程具有多个维度时 $p(X) \neq p(c_1 X_1 + c_2 X_2 + \dots + b)$，如何设计代码框架利用 Pearson 相关系数对空间进行筛选？
2. 如何设计算法以增加结果的可解释性？（提示：实际中处理的问题存在物理背景，各个输入有其物理含义）
3. 如何将神经网络或者大语言模型这种黑盒子与符号回归方法整合到一起以增强其可解释性与搜索能力？