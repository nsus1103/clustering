import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools


def isCore(X, i, pts, minPts, eps):
    n = Neighbors(X, i, pts, eps)
    if len(n) >= minPts - 1:
        return True, n
    else:
        return False, n


def Neighbors(X, i, pts, eps):
    neighbor_names = []
    neighbor_points = []
    for j in range(len(XX)):
        dist = compute_distances(X[i], XX[j])

        if dist < eps and dist > 0:
            neighbor_names.append(X_labels[j])
            neighbor_points.append(XX[j])

    return neighbor_names


def compute_distances(pt1, pt2):
    distance = np.linalg.norm(pt1 - pt2)
    return distance


def dbscan(X, labels, minPts, eps, k):
    for i in range(len(X)):
        if labels[i] not in classified:
            print(len(X))
            is_core, n = isCore(X, i, labels, minPts, eps)

            if is_core:
                print(f"adding {labels[i]} to core")
                print(f"neighbours of {labels[i]} is", n)
                c.append(labels[i])
                for cahr in n:
                    core_n.append(cahr)
                classified.append(labels[i])
                core_X = np.array([points[a] for a in n])
                dbscan(core_X, n, minPts, eps, k)
                clustering[k] = c
            else:
                print(f"adding {labels[i]} to whatever")
                if labels[i] in core_n:
                    print(f"adding {labels[i]} to border")
                    print(f"neighbours of {labels[i]} is", n)
                    b.append(labels[i])
                else:
                    print(f"adding {labels[i]} to noise")
                    print(f"neighbours of {labels[i]} is", n)
                    noise.append(labels[i])
                classified.append(labels[i])
            k += 1
    return c, b, noise


points = {'A':(2,3),
          'B':(7,9),
          'C':(11,4),
          'D':(3,3),
          'E':(14,12),
          'F':(4,5),
          'G':(12,5),
          'H':(9,7),
          'J':(6,4),
          'K':(2,9),
          'L':(4,7),
          'M':(6,8)}
XX = np.array(list(points.values()))
X_labels = np.array(list(points.keys()))

eps=3
minPts = 3
c=[]
b=[]
noise=[]
core_n=[]
classified=[]
clustering={}
k=1
co,bo,no = dbscan(XX, X_labels, minPts, eps, k)

corepts = np.array([points[a] for a in co])
borderpts = np.array([points[a] for a in bo])
noisepts = np.array([points[a] for a in no])
clusterk = [corepts, borderpts, noisepts]

colors = iter(cm.rainbow(np.linspace(0, 1, len(clustering))))
labels = ['core points', 'border points', 'noise points']
i = 0
for cluster in clusterk:
    if len(cluster) > 0:
        a = plt.scatter(cluster[:, 0], cluster[:, 1], color=next(colors), label=labels[i])
        plt.legend()
    i += 1

plt.title('distribution of clustered points')
for i in range(len(XX)):
    plt.text(y=XX[i][1] + 0.2, x=XX[i][0] + 0.2, s=X_labels[i])

plt.show()

# using Sklean.cluster.DBSCAN

from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=3, min_samples=3, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None,
             n_jobs=None)
dbs.fit(XX)
print(dbs.labels_) # -1 denotes noise