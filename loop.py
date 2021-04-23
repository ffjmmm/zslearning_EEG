from PyNomaly import loop
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

iris = pd.DataFrame(data('iris'))
iris = pd.DataFrame(iris.drop('Species', 1))

scores = loop.LocalOutlierProbability(iris).fit().local_outlier_probabilities

iris['scores'] = scores

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iris['Sepal.Width'], iris['Petal.Width'], iris['Sepal.Length'],
           c=iris['scores'], cmap='seismic', s=50)
ax.set_xlabel('Sepal.Width')
ax.set_ylabel('Petal.Width')
ax.set_zlabel('Sepal.Length')
plt.show()
plt.clf()
plt.cla()
plt.close()