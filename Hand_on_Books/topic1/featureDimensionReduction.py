from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2
from sklearn.datasets import load_iris
from scipy.stats import pearsonr
from array import array
from minepy import MINE
import numpy as np
iris = load_iris()
# 方差选择法
# 参数threshold 为方差的阈值
VarianceThreshold(threshold=3).fit_transform(iris.data)

"""SelectKBest 's three method"""
# 1.相关系数法
#选择K个最好的特征，返回特征数据
# 第一个参数为计算评估特征的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组， 数组
# 第i项为第i个特征的评分和P值，在此定义为计算相关系数
# 参数k为选择的特征个数
SelectKBest(
    lambda X, Y:np.array(list(map(lambda x:pearsonr(x, Y), X.T))).T[0], k=2
).fit_transform(iris.data, iris.target)

# 2.卡方检验 经典的卡方检验是检验定性自变量和定性因变量的相关性
SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)

# 3.最大信息系数法
# 由于MINE的设计不是函数式的，因此需要定义mic方法将其转换为函数式，返回一个二元组，二元组的第二项设置
# 成固定的P值， 为0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)

# 选择k个最好的特征， 返回特征选择后的数据
SelectKBest(lambda X, Y:np.array(list(map(lambda x:mic(x, Y), X.T))).T[0],
            k=2).fit_transform(iris.data, iris.target)