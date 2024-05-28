import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

data = np.array([[-37.530, 3.109, -16.452],
                [40.247, 5.483, -15.209],
                [-31.920, 12.584, -12.916],
                [-32.760, 14.072, -13.749],
                [-37.100, 1.953, -15.720],
                [-32.143, 12.990, -13.488],
                [-41.077, 4.651, -15.651], 
                [-34.219, 13.611, -13.090],
                [-33.117, 15.875, -13.738]])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:,0], data[:,1], data[:,2], s=300)
ax.view_init(azim=200)
plt.show()

model = DBSCAN(eps=2.5, min_samples=2)
model.fit_predict(data)
pred = model.fit_predict(data)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:,0], data[:,1], data[:,2], c=model.labels_, s=300)
ax.view_init(azim=200)
plt.show()

print("number of cluster found: {}".format(len(set(model.labels_))))
print('cluster for each point: ', model.labels_)