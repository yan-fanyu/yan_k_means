import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 读取文件
datafile = u'data.xlsx' # 文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
outfile = 'stu2.xlsx'
data = pd.read_excel(datafile)  # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
d = DataFrame(data)

# 聚类
n = 5                         # 聚成 5 类数据
mod = KMeans(n_clusters=n)
mod.fit_predict(d)  # y_pred表示聚类的结果

# 聚成 5 类数据，统计每个聚类下的数据量，并且求出他们的中心
r1 = pd.Series(mod.labels_).value_counts()  # 每个类下面有多少个样本
r2 = pd.DataFrame(mod.cluster_centers_)     # 中心
r = pd.concat([r2, r1], axis=1)
r.columns = list(d.columns) + [u'类别数目']


# 给每一条数据标注上被分为哪一类
r = pd.concat([d, pd.Series(mod.labels_, index=d.index)], axis=1)
r.columns = list(d.columns) + [u'聚类类别']
print(r)
r.to_excel(outfile)  # 如果需要保存到本地，就写上这一列

# 可视化过程

ts = TSNE()
ts.fit_transform(r)
ts = pd.DataFrame(ts.embedding_, index=r.index)

a = ts[r[u'聚类类别'] == 0]
plt.plot(a[0], a[1], 'r.')
a = ts[r[u'聚类类别'] == 1]
plt.plot(a[0], a[1], 'go')
a = ts[r[u'聚类类别'] == 2]
plt.plot(a[0], a[1], 'g*')
a = ts[r[u'聚类类别'] == 3]
plt.plot(a[0], a[1], 'b.')
a = ts[r[u'聚类类别'] == 4]
plt.plot(a[0], a[1], 'b*')
plt.show()