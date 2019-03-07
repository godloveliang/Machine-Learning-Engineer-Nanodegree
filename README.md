# 机器学习纳米学位

---

# 项目1：模型评估与验证

## 波士顿房价预测

### 准备工作

这个项目需要安装**Python 2.7**和以下的Python函数库：

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

你还需要安装一个软件，以运行和编辑[ipynb](http://jupyter.org/)文件。

优达学城推荐学生安装 [Anaconda](https://www.continuum.io/downloads)，这是一个常用的Python集成编译环境，且已包含了本项目中所需的全部函数库。我们在P0项目中也有讲解[如何搭建学习环境](https://github.com/nd009/titanic_survival_exploration/blob/master/README.md)。

### 代码

代码的模版已经在`boston_housing.ipynb`文件中给出。你还会用到`visuals.py`和名为`housing.csv`的数据文件来完成这个项目。我们已经为你提供了一部分代码，但还有些功能需要你来实现才能以完成这个项目。

### 运行

在终端或命令行窗口中，选定`boston_housing/`的目录下（包含此README文件），运行下方的命令：

```jupyter notebook boston_housing.ipynb```

这样就能够启动jupyter notebook软件，并在你的浏览器中打开文件。

### 数据

经过编辑的波士顿房价数据集有490个数据点，每个点有三个特征。这个数据集编辑自[加州大学欧文分校机器学习数据集库（数据集已下线）](https://archive.ics.uci.edu/ml/datasets.html).

**特征**

1. `RM`: 住宅平均房间数量
2. `LSTAT`: 区域中被认为是低收入阶层的比率
3. `PTRATIO`: 镇上学生与教师数量比例

**目标变量**

4. `MEDV`: 房屋的中值价格

---
# 项目2: 为CharityML寻找捐献者

## 监督学习

### 安装

这个项目要求使用 Python 2.7 并且需要安装下面这些python包：

- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)

你同样需要安装好相应软件使之能够运行 [iPython Notebook](http://ipython.org/notebook.html)

优达学城推荐学生安装[Anaconda](https://www.continuum.io/downloads), 这是一个已经打包好的python发行版，它包含了我们这个项目需要的所有的库和软件。

### 代码

初始代码包含在`finding_donors.ipynb`这个notebook文件中。你还会用到`visuals.py`和名为`census.csv`的数据文件来完成这个项目。我们已经为你提供了一部分代码，但还有些功能需要你来实现才能以完成这个项目。
这里面有一些代码已经实现好来帮助你开始项目，但是为了完成项目，你还需要实现附加的功能。  
注意包含在`visuals.py`中的代码设计成一个外部导入的功能，而不是打算学生去修改。如果你对notebook中创建的可视化感兴趣，你也可以去查看这些代码。


### 运行
在命令行中，确保当前目录为 `finding_donors/` 文件夹的最顶层（目录包含本 README 文件），运行下列命令：

```bash
jupyter notebook finding_donors.ipynb
```

​这会启动 Jupyter Notebook 并把项目文件打开在你的浏览器中。

### 数据

修改的人口普查数据集含有将近32,000个数据点，每一个数据点含有13个特征。这个数据集是Ron Kohavi的论文*"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",*中数据集的一个修改版本。你能够在[这里](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf)找到论文，在[UCI的网站](https://archive.ics.uci.edu/ml/datasets/Census+Income)找到原始数据集。

**特征**

- `age`: 一个整数，表示被调查者的年龄。 
- `workclass`: 一个类别变量表示被调查者的通常劳动类型，允许的值有 {Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked}
- `education_level`: 一个类别变量表示教育程度，允许的值有 {Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool}
- `education-num`: 一个整数表示在学校学习了多少年 
- `marital-status`: 一个类别变量，允许的值有 {Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse} 
- `occupation`: 一个类别变量表示一般的职业领域，允许的值有 {Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces}
- `relationship`: 一个类别变量表示家庭情况，允许的值有 {Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried}
- `race`: 一个类别变量表示人种，允许的值有 {White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black} 
- `sex`: 一个类别变量表示性别，允许的值有 {Female, Male} 
- `capital-gain`: 连续值。 
- `capital-loss`: 连续值。 
- `hours-per-week`: 连续值。 
- `native-country`: 一个类别变量表示原始的国家，允许的值有 {United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands}

**目标变量**

- `income`: 一个类别变量，表示收入属于那个类别，允许的值有 {<=50K, >50K}

---
# 项目 3: 创建用户细分

## 非监督学习

### 安装

这个项目要求使用 **Python 2.7** 并且需要安装下面这些python包：

- [NumPy](http：//www.numpy.org/)
- [Pandas](http：//pandas.pydata.org)
- [scikit-learn](http：//scikit-learn.org/stable/)

你同样需要安装好相应软件使之能够运行[Jupyter Notebook](http://jupyter.org/)。

优达学城推荐学生安装 [Anaconda](https：//www.continuum.io/downloads), 这是一个已经打包好的python发行版，它包含了我们这个项目需要的所有的库和软件。

### 代码

初始代码包含在 `customer_segments.ipynb` 这个notebook文件中。这里面有一些代码已经实现好来帮助你开始项目，但是为了完成项目，你还需要实现附加的功能。

### 运行

在命令行中，确保当前目录为 `customer_segments.ipynb` 文件夹的最顶层（目录包含本 README 文件），运行下列命令：

```jupyter notebook customer_segments.ipynb```

​这会启动 Jupyter Notebook 并把项目文件打开在你的浏览器中。

## 数据

​这个项目的数据包含在 `customers.csv` 文件中。你能在[UCI 机器学习信息库](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)页面中找到更多信息。

---
# Project4: Train a Smartcab How to Drive

## Reinforcement Learning

### Install

This project requires **Python 2.7** with the [pygame](https://www.pygame.org/wiki/GettingStarted
) library installed

### Code

Template code is provided in the `smartcab/agent.py` python file. Additional supporting python code can be found in `smartcab/enviroment.py`, `smartcab/planner.py`, and `smartcab/simulator.py`. Supporting images for the graphical user interface can be found in the `images` folder. While some code has already been implemented to get you started, you will need to implement additional functionality for the `LearningAgent` class in `agent.py` when requested to successfully complete the project. 

### Run

In a terminal or command window, navigate to the top-level project directory `smartcab/` (that contains this README) and run one of the following commands:

```python smartcab/agent.py```  
```python -m smartcab.agent```

This will run the `agent.py` file and execute your agent code.

---
# 项目  5: 图像分类项目

## 深度学习


在此项目中，你将对 [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html) 中的图片进行分类。该数据集包含飞机、猫狗和其他物体。你需要预处理这些图片，然后用所有样本训练一个卷积神经网络。图片需要标准化（normalized），标签需要采用 one-hot 编码。你需要应用所学的知识构建卷积的、最大池化（max pooling）、丢弃（dropout）和完全连接（fully connected）的层。最后，你需要在样本图片上看到神经网络的预测结果。

由于本项目需要耗费大量CPU，如有独立显卡的同学可以选择下载到本地调试运行，其他同学推荐使用AWS的远程服务器。再此推荐我们的学霸reviewer杨培文撰写的[如何在AWS上配置深度学习主机](https://zhuanlan.zhihu.com/p/25066187)。配置完成后，即可使用AWS的主机来进行代码的运算及调试。
