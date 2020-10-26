from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.preprocessing import Normalizer, Binarizer,OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from numpy import vstack, array, nan, log1p

iris = load_iris()

def standard_scalar():
    # 标准化， 返回值为标准化后的数据
    res = StandardScaler().fit_transform(iris.data)
    print(res)
# standard_scalar()
def minmax_scalar():
    # 区间缩放，返回值为缩放到[0, 1]区间的数据
    MinMaxScaler().fit_transform(iris.data)

def normalizer():
    # 归一化 ， 把数据映射到[0, 1]或者[a, b]区间内
    Normalizer().fit_transform(iris.data)

def binarizer():
    # 定量特征二值化 设定一个阈值，大于阈值为1，小于等于为0
    Binarizer(threshold=3).fit_transform(iris.data)

def one_hot_encoder():
    # 定性特征哑编码
    OneHotEncoder(categories='auto').fit_transform(iris.target.reshape((-1, 1)))

def deal_nan():
    # 缺失值处理， 返回值为处理缺失值后的数据
    # 参数 missing_value 为缺失值的表示形式， 默认为NaN
    # 参数strategy 为缺失值的填充方式， 默认为mean(均值)
    res = SimpleImputer().fit_transform(vstack((array([nan, nan, nan, nan]),
                                          iris.data)))
    print(res)
# deal_nan()

def data_exchange():
    # 多项式转换
    # 参数 degree 为度
    PolynomialFeatures().fit_transform(iris.data)
    # 对数变换
    # 自定义转换函数为对数函数的数据变换
    # 第一个参数是单变元函数
    FunctionTransformer(log1p, validate=False).fit_transform(iris.data)