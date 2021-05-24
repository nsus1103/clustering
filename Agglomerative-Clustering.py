# A (2, 3), B (7,9), C(11,4), D(3,3), E(14,12), F(4,5), G(12, 5), H(9, 7), J(6,4), K(2,9), L(4,7), M(6,8)
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

def calculate_distances(clustering, linkage='single'):

    dist_matrix = np.diag(np.ones(len(clustering)))

    for i in range(len(clustering)):
        for j in range(len(clustering)):
            if i!=j:
                dist_matrix[i][j] = proximity(clustering[i], clustering[j], linkage)
            else:
                dist_matrix[i][j] = 1000

    return dist_matrix

def proximity(sample1, sample2, linkage):
    dist=[]
    for i in range(len(sample1)):
        for j in range(len(sample2)):
            try:
                dist.append(np.linalg.norm(np.array(sample1[i]) - np.array(sample2[j])))
            except:
                print('Here it dies:sample1', len(sample1[i]),'sample2:', sample2[j])
                # print(type(sample1), type(sample2))
                # print(len(sample1), len(sample2))

    if linkage=='complete':
        return max(dist)
    elif linkage=='single':
        return min(dist)
    else:
        return np.mean(dist)

#

# points = {'A':(2,3),
#           'B':(7,9),
#           'C':(11,4),
#           'D':(3,3),
#           'E':(14,12),
#           'F':(4,5),
#           'G':(12,5),
#           'H':(9,7),
#           'J':(6,4),
#           'K':(2,9),
#           'L':(4,7),
#           'M':(6,8)}

points = {'A':(-9,0),
          'B':(-1,0),
          'C':(5,0),
          'D':(7,0),
          'E':(10,0),
          'F':(14,0),
          'G':(21,0),
          'H':(24,0),
          'J':(26,0),
          'K':(32,0),
          'L':(33,0),
          'M':(64,0)}
# X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])

X = np.array(list(points.values()))
# X=np.array([-9, -1, 5, 7 , 10, 14, 21, 24, 26, 32, 33, 64])

l = list(points.keys())
locs = [[l[i]] for i in range (X.shape[0])]
clustering = [[X[i]] for i in range(X.shape[0])]
c = len(clustering)

clusters={}
k=1
test={}
while c>1:
    print(k)
    # Compute proximity matrix for the clusters
    proxm = calculate_distances(clustering, linkage='complete')
    # proxm is a diagonal matrix therefore there would be two index for each min, keeping first
    # min_indexes = np.where(proxm==proxm.min())[0]
    # min_indexes = np.array((np.where(proxm==proxm.min())[0][0],np.where(proxm==proxm.min())[1][0]))
    min_indexes = np.array((np.where(proxm==proxm.min())[0][0],np.where(proxm==proxm.min())[1][0]))

    clusters[f'C{k}']=[locs[min_indexes[0]].copy(), locs[min_indexes[1]].copy(), proxm[min_indexes[0]][min_indexes[1]].copy()]
    print('min1:',locs[min_indexes[0]],'min2:',locs[min_indexes[1]])

    """ min_index comprises two index, we take the second and insert it
    along with the first creating a new list inside the list"""
    c2 = clustering.pop(min_indexes[1])
    clustering[min_indexes[0]].append(c2)
    # clustering[min_indexes[0]] = [clustering[min_indexes[0]]]

    f = locs.pop(min_indexes[1])
    locs[min_indexes[0]].append(f)
    # locs[min_indexes[0]] = [locs[min_indexes[0]]]

    c=len(clustering)
    # c=1
    k+=1



#
#Plotting the dendrogram
Z = linkage(X, 'complete')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()

#
model = AgglomerativeClustering(n_clusters=1, affinity='euclidean', compute_full_tree=True, linkage='complete', distance_threshold=None)
model.fit(X)


