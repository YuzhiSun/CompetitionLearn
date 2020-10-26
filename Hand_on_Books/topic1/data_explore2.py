from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


train_data_file = "D:\\project\\TianChi\\data\\zhengqi_train.txt"
test_data_file = "D:\\project\\TianChi\\data\\zhengqi_test.txt"

train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')


X_train = train_data.iloc[:, 0:-1]
y_train = train_data.iloc[:, -1]


def find_outliers(model, X, y, sigma=3):
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    print('R2=', model.score(X, y))
    print("mse=", mean_squared_error(y, y_pred))
    print('-----------------------------------------')
    print('mean of residuals：', mean_resid)
    print('std of residuals:', std_resid)
    print('-----------------------------------------')
    print(len(outliers), 'outliers:')
    print(outliers.tolist())

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.' )
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax = ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('z')
    plt.savefig('D:\\project\\TianChi\\Hand_on_Books\\topic1\\outliers.png')
    return outliers

# outliers = find_outliers(Ridge(), X_train, y_train)
def draw_QQ_sample():
    plt.figure(figsize=(10, 5))

    ax = plt.subplot(1,2,1)
    sns.distplot(train_data['V0'], fit = stats.norm)
    ax = plt.subplot(1,2,2)
    res = stats.probplot(train_data['V0'], plot = plt)
    # plt.show()

def draw_QQ_of_variable():
    train_cols = 6
    train_rows = len(train_data.columns)
    plt.figure(figsize=(4 * train_cols, 4 * train_rows))

    i = 0

    for col in train_data.columns:
        i+=1
        ax = plt.subplot(train_rows, train_cols, i)
        sns.distplot(train_data[col], fit=stats.norm)

        i+=1
        ax = plt.subplot(train_rows, train_cols, i)
        res = stats.probplot(train_data[col], plot = plt)

    plt.tight_layout()
    plt.savefig('D:\\project\\TianChi\\Hand_on_Books\\topic1\\QQ.png')
# draw_QQ_of_variable()

def KDE():
    plt.figure(figsize=(8, 4), dpi=150)
    ax = sns.kdeplot(train_data['V0'], color="Red", shade=True)
    ax = sns.kdeplot(test_data['V0'], color="Blue", shade=True)
    ax.set_xlabel('V0')
    ax.set_ylabel("Frequency")
    ax = ax.legend(["train", "test"])
    # plt.show()
    dist_cols = 6
    dist_rows = len(test_data.columns)
    plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
    i = 1
    for col in test_data.columns:
        ax = plt.subplot(dist_rows, dist_cols, i)
        ax = sns.kdeplot(train_data[col], color="Red", shade=True)
        ax = sns.kdeplot(test_data[col], color="Blue", shade=True)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax = ax.legend(["train", "test"])
        i += 1
    plt.savefig('D:\\project\\TianChi\\Hand_on_Books\\topic1\\KDE.png')
# KDE()

def liner_reg():
    fcols = 2
    frows = 1
    plt.figure(figsize=(8, 4), dpi=150)

    ax = plt.subplot(1,2,1)
    sns.regplot(x='V0', y='target', data=train_data, ax=ax,
                scatter_kws={'marker':'.', 's':3, 'alpha':0.3},
                line_kws={'color':'k'})

    plt.xlabel('V0')
    plt.ylabel('target')

    ax = plt.subplot(1,2,2)
    sns.distplot(train_data['V0'].dropna())
    plt.xlabel('V0')

    # plt.show()

    fcols = 6
    frows = len(test_data.columns)
    plt.figure(figsize=(5 * fcols, 4 * frows))

    i = 0
    for col in test_data.columns:
        i += 1
        ax = plt.subplot(frows, fcols, i)
        sns.regplot(x=col, y='target', data=train_data, ax = ax,
                    scatter_kws={'marker':'.', 's':3, 'alpha':0.3},
                    line_kws={'color':'k'})
        plt.xlabel(col)
        plt.ylabel('target')

        i += 1

        ax = plt.subplot(frows, fcols, i)
        sns.distplot(train_data[col].dropna())
        plt.xlabel(col)
    plt.savefig('D:\\project\\TianChi\\CompetitionLearn\\Hand_on_Books\\topic1\\image\\line_reg.png')

# liner_reg()

data_train1 = train_data.drop(['V5','V9','V11','V17','V22','V28'], axis=1)

def cal_corr():
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 10)
    # 删掉测试集和训练集分布不一致的变量 得到 data_train1  在函数外已经执行
    train_corr = data_train1.corr()
    # print(train_corr)

    # 画热力图
    ax = plt.subplots(figsize=(20, 16))
    ax = sns.heatmap(train_corr, vmax=.8, square=True, annot=True)
    # plt.show(annot=True)  # annot=True 表示显示系数

    # 根据相关系数筛选特征变量
    k = 10  # 需要筛选出的特征数量
    cols = train_corr.nlargest(k, 'target')['target'].index  #按照指定列选出前n大的行

    cm = np.corrcoef(train_data[cols].values.T)
    hm = plt.subplots(figsize = (10, 10))
    hm = sns.heatmap(train_data[cols].corr(), annot=True, square=True)
    plt.show()
    # 找出相关系数大于0.5的变量
    threshold = 0.5

    corrmat = train_data.corr()
    top_corr__features = corrmat.index[abs(corrmat['target']) > threshold]
    plt.figure(figsize=(10, 10))
    g = sns.heatmap(train_data[top_corr__features].corr(),
                    annot=True,
                    cmap="RdYlGn")
    # plt.show()
    # 用相关系数阈值移除相关特征
    corr_matrix = data_train1.corr().abs()
    drop_col = corr_matrix[corr_matrix['target'] < threshold].index
    # data_all.drop(drop_col, axis=1, inplace=True)

# cal_corr()

def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())

def Box_Cox():
    # Box-Cox 变换
    drop_columns = ['V5', 'V9', 'V11', 'V17', 'V22', 'V28']

    # 合并训练集和测试集的数据
    train_x = train_data.drop(['target'], axis=1)

    data_all = pd.concat([train_x, test_data])

    data_all.drop(drop_columns, axis=1, inplace=True)
    # print(data_all.head())

    cols_numeric = list(data_all.columns)
    # 对每列数据进行归一化
    data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis=0)
    # print(data_all[cols_numeric].describe())
    train_data_process = train_data[cols_numeric]
    train_data_process = train_data_process[cols_numeric].apply(scale_minmax,
                                                                axis=0)

    test_data_process = test_data[cols_numeric]
    test_data_process = test_data_process[cols_numeric].apply(scale_minmax,
                                                              axis=0)
    # 变换后， 计算分位数并画图展示，显示特征变量与target变量的线性关系
    cols_numeric_left = cols_numeric[0:13]
    cols_numeric_right = cols_numeric[13:]
    train_data_process = pd.concat([train_data_process, train_data['target']],
                                   axis=1)
    fcols = 6
    frows = len(cols_numeric_left)
    plt.figure(figsize=(4 * fcols, 4 * frows))
    i = 0

    for var in cols_numeric_left:
        dat = train_data_process[[var,'target']].dropna()
        i += 1
        plt.subplot(frows, fcols, i)
        sns.distplot(dat[var], fit=stats.norm)
        plt.title(var + ' Original')
        plt.xlabel('')

        i += 1
        plt.subplot(frows, fcols, i)
        _ = stats.probplot(dat[var], plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[var])))
        plt.xlabel('')
        plt.ylabel('')
        i += 1
        plt.subplot(frows, fcols, i)
        plt.plot(dat[var], dat['target'], '.', alpha=0.5)
        plt.title('corr=' +
                  '{:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1]))
        i += 1
        plt.subplot(frows, fcols, i)
        trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
        trans_var = scale_minmax(trans_var)
        sns.distplot(trans_var, fit=stats.norm)
        plt.title(var + ' Transformed')
        plt.xlabel('')
        i += 1
        plt.subplot(frows, fcols, i)
        _ = stats.probplot(trans_var, plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(trans_var)))
        plt.xlabel('')
        plt.ylabel('')
        i += 1
        plt.subplot(frows, fcols, i)
        plt.plot(trans_var, dat['target'], '.', alpha=0.5)
        plt.title('corr=' +
                  ':.2f'.format(np.corrcoef(trans_var, dat['target'])[0][1]))
        plt.savefig('D:\\project\\TianChi\\CompetitionLearn\\Hand_on_Books\\topic1\\image\\Box-Cox.png')

# Box_Cox()
