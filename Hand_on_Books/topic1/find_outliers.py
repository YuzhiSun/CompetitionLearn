from scipy import  stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
KDE()