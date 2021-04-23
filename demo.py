from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
x = [i / 10. for i in range(10)]
for a in x:
    ax.scatter(a, a, c='black', alpha=a)
plt.show()