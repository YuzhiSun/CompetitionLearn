#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import  stats

import warnings

from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

train_data_file = "D:\\project\\TianChi\\data\\zhengqi_train.txt"
test_data_file = "D:\\project\\TianChi\\data\\zhengqi_test.txt"

train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

train_data.info()
test_data.info()

train_data.describe()

test_data.describe()

train_data.head(10)

fig = plt.figure(figsize=(4, 6))
sns.boxplot(train_data['V0'], orient="v", width=0.5)
plt.show()
column = train_data.columns.tolist()[:39]
fig = plt.figure(figsize=(80,60), dpi=75)
for i in range(38):
    plt.subplot(5, 8, i + 1)
    sns.boxplot(train_data[column[i]], orient="v", width=0.5)
    plt.ylabel(column[i], fontsize=36)

plt.savefig('D:\\project\\TianChi\\Hand_on_Books\\topic1\\box.png')



