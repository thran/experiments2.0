import numpy as np
import math
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform


class KMeanPrecomputed:
    def __init__(self, distance_matrix):
        self.distances = distance_matrix
        self.point_count = distance_matrix.shape[0]

        self.nearest_centroids = None
        self.centroids = None
        self.cluster_count = 0
        self.cost = 0

    def run(self, cluster_count=2, max_iterations=1000):
        self.cluster_count = cluster_count

        self.clusters_init()
        last_cost = 0
        self.clusters_match()
        for i in range(max_iterations):
            self.clusters_move()
            print(1, self.cluster_cost())
            self.clusters_match()
            print(2, self.cluster_cost())

            self.cost = self.cluster_cost()

            if last_cost == self.cost:
                break
            last_cost = self.cost

        return self.nearest_centroids

    def clusters_init(self):
        self.centroids = np.random.choice(self.point_count, self.cluster_count)
        self.nearest_centroids = np.random.choice(self.cluster_count, self.point_count)

    def clusters_match(self):
        self.nearest_centroids = self.distances[:, self.centroids].argmin(axis=1)

    def clusters_move(self):
        for i, centroid in enumerate(self.centroids):
            points_in_cluster = self.nearest_centroids == i
            new_centroid_index = self.distances[points_in_cluster][:, points_in_cluster].sum(axis=0).argmin()
            new_centroid = np.argwhere(points_in_cluster.cumsum() == new_centroid_index + 1)[0][0]
            self.centroids[i] = new_centroid

    def cluster_cost(self):
        return self.distances[range(self.point_count), self.centroids[self.nearest_centroids]].sum() / self.point_count


def demo():
    dat = 1000
    a = np.empty((4*dat,2),float)
    for i in range(dat):
        a[i] = np.random.multivariate_normal([0,0],[[1,0.1],[1,.7]])
        a[i+dat] = np.random.multivariate_normal([3,2],[[0.5,0],[0,2]])
        a[i+2*dat] = np.random.multivariate_normal([3,2],[[0.5,0],[0,2]])
        a[i+3*dat] = np.random.multivariate_normal([0,4],[[0.1,0],[0,.1]])

    distance_matrix = squareform(pdist(a, 'euclidean'))
    km = KMeanPrecomputed(distance_matrix)
    labels = km.run(3)
    z = (labels == 2)
    x = (labels == 1)
    y = (labels == 0)


    plt.plot(a[x,0],a[x,1],"ro")
    plt.plot(a[y,0],a[y,1],"bo")
    plt.plot(a[z, 0],a[z, 1],"go")
    plt.show()

# demo()
