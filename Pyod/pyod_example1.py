import pandas as pd
import numpy as np
from scipy import stats
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm

df = pd.read_csv("train.csv")
print(df.describe())
show = False
if show:
    plt.figure(figsize=(10, 10))
    plt.scatter(df['Item_MRP'], df['Item_Outlet_Sales'])
    plt.xlabel("Item_MRF")
    plt.ylabel("Item_Outlet_Sales")
    plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
df[['Item_MRP', 'Item_Outlet_Sales']] = scaler.fit_transform(df[['Item_MRP', 'Item_Outlet_Sales']])
print(df[['Item_MRP', 'Item_Outlet_Sales']].head())

x1 = df['Item_MRP'].values.reshape(-1, 1)
x2 = df['Item_Outlet_Sales'].values.reshape(-1, 1)
x = np.concatenate((x1, x2), axis=1)
# 设置 5%的离群点数据
random_state = np.random.RandomState(42)
outliers_fraction = 0.05
# 定义7个后续会使用的离群点检测模型
classifiers = {
    "Angle-based Outlier Detector(ABOD)": ABOD(contamination=outliers_fraction),
    "Cluster-based Local Outiler Factor (CBLOF)": CBLOF(contamination=outliers_fraction, check_estimator=False,
                                                        random_state=random_state),
    "Feature Bagging": FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction, check_estimator=False,
                                      random_state=random_state),
    "Histogram-base Outlier Detection(HBOS)": HBOS(contamination=outliers_fraction),
    "Isolation Forest": IForest(contamination=outliers_fraction, random_state=random_state),
    "KNN": KNN(contamination=outliers_fraction),
    "Average KNN": KNN(method='mean', contamination=outliers_fraction)
}

# 逐一 比较模型
xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(x)
    # 预测利群得分
    scores_pred = clf.decision_function(x) * -1
    # 预测数据点是否为 离群点
    y_pred = clf.predict(x)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))

    # 复制一份数据
    dfx = df
    dfx['outlier'] = y_pred.tolist()
    # IX1 非离群点的特征1，IX2 非利群点的特征2
    IX1 = np.array(dfx['Item_MRP'][dfx['outlier'] == 0]).reshape(-1, 1)
    IX2 = np.array(dfx['Item_Outlet_Sales'][dfx['outlier'] == 0]).reshape(-1, 1)
    # OX1 离群点的特征1，OX2离群点特征2
    OX1 = np.array(dfx['Item_MRP'][dfx['outlier'] == 1]).reshape(-1, 1)
    OX2 = np.array(dfx['Item_Outlet_Sales'][dfx['outlier'] == 1]).reshape(-1, 1)
    print("模型 %s 检测到的" % clf_name, "离群点有 ", n_outliers, "非离群点有", n_inliers)

    # 判定数据点是否为离群点的 阈值
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
    # 决策函数来计算原始的每个数据点的离群点得分
    z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    z = z.reshape(xx.shape)
    # 最小离群得分和阈值之间的点 使用蓝色填充
    plt.contourf(xx, yy, z, levels=np.linspace(z.min(), threshold, 7), cmap=plt.cm.Blues_r)
    # 离群得分等于阈值的数据点 使用红色填充
    a = plt.contour(xx, yy, z, levels=[threshold], linewidths=2, colors='red')
    # 离群得分在阈值和最大离群得分之间的数据 使用橘色填充
    plt.contourf(xx, yy, z, levels=[threshold, z.max()], colors='orange')
    b = plt.scatter(IX1, IX2, c='white', s=20, edgecolor='k')
    c = plt.scatter(OX1, OX2, c='black', s=20, edgecolor='k')
    plt.axis('tight')
    # loc = 2 用来左上角
    plt.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'inliers', 'outliers'],
        prop=mfm.FontProperties(size=20),
        loc=2
    )
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(clf_name)
    plt.savefig("%s.png" % clf_name)
    # plt.show()
