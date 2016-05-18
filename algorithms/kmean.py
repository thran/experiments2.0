import numpy as np
import random
import math
import pylab as plt
import scipy.stats as stats


def distance(x, y):
    d = 0
    for i in range(np.size(x)):
        d += abs(y[i] - x[i]) ** 2
    return math.sqrt(d)


class KMean:
    def __init__(self, points):
        self.points = points
        self.point_count = points.shape[0]
        self.feature_count = points.shape[1]

        self.nearest_centroids = None
        self.centroids = None
        self.cluster_count = 0
        self.cost = 0

    def run(self, cluster_count=2, max_iterations=50):
        self.cluster_count = cluster_count

        self.clusters_init()
        last_cost = 0
        for i in range(max_iterations):
            self.clusters_match()
            self.clusters_move()
            self.cost = self.clusterCost()

            if last_cost == self.cost:
                break
            last_cost = self.cost

        return self.nearest_centroids

    def clusters_init(self):
        self.nearest_centroids = np.empty(self.point_count, int)
        self.centroids = self.points[np.random.choice(self.point_count, self.cluster_count)]

    def clusters_match(self):
        for i, point in enumerate(self.points):
            best_distance = np.inf
            for j, centroid in enumerate(self.centroids):
                d = self.distance(point, centroid)
                if d < best_distance:
                    best_distance = d
                    self.nearest_centroids[i] = j

    def clusters_move(self):
        for c in range(self.cluster_count):
            m = self.points[(self.nearest_centroids == c), :]
            m = np.ma.masked_array(m, np.isnan(m))
            self.centroids_positions[c, :] = np.mean(m, 0)

    def cluster_cost(self):
        c = 0
        for p in range(self.point_count):
            c += distance(self.points[p, :], self.centroids_positions[self.nearest_centroids[p]])
        return c / self.point_count
