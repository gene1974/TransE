import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import dbscan

def cluster_data(data):
    df = pd.DataFrame(X,columns = ['feature1','feature2'])
    core_samples,cluster_ids = dbscan(X, eps = 0.2, min_samples=20) 
    print()

X,_ = datasets.make_moons(500,noise = 0.1,random_state=1) # (500, 2)
df = pd.DataFrame(X,columns = ['feature1','feature2'])

# df.plot.scatter('feature1','feature2', s = 100,alpha = 0.6, title = 'dataset by make_moon')

# eps为邻域半径，min_samples为最少点数目
core_samples,cluster_ids = dbscan(X, eps = 0.2, min_samples=20) 
print(cluster_ids.shape)
# cluster_ids中-1表示对应的点为噪声点

df = pd.DataFrame(np.c_[X,cluster_ids],columns = ['feature1','feature2','cluster_id'])
df['cluster_id'] = df['cluster_id'].astype('i2')

# df.plot.scatter('feature1','feature2', s = 100,
#     c = list(df['cluster_id']),cmap = 'rainbow',colorbar = False,
#     alpha = 0.6,title = 'sklearn DBSCAN cluster result')

plt.scatter(df["feature1"], df["feature2"], c = list(df['cluster_id']), cmap = 'rainbow', marker = '.')
# plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
plt.xlabel("Calories")
plt.ylabel("Alcohol")
plt.savefig('cluster.png')



