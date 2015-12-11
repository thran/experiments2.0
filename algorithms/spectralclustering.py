from algorithms.kmean import *
import numpy as np
import numpy.linalg as la
import sys
import heapq
import scipy.cluster.vq as clust

def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                yield list(perm[:i]) + list(elements[0:1]) + list(perm[i:])

class SpectralClusterer:
    def __init__(self, corr, kcut=0, mutual=True):         #square matrix with values in <0,inf> 0 - far, inf - close
        self.corr = np.copy(corr)

        if (self.corr.shape[0]!=self.corr.shape[1]):
            raise ValueError("Data metrix", "Data metrix must be square, but shape is "+str(data.shape))

        self.W = self.corr
        self.point_number = self.corr.shape[0]
        for i in range(self.point_number):
            self.W[i,i] = 0

        if kcut>0:
            if mutual:
                self._kNearestMutual(kcut)
            else:
                self._kNearest(kcut)

    def run(self, cluster_number=2, sc_type=0, KMiter=20, KMlib=True):
        """
            sc_type = 0   -   Unnormalized spectral clustering
            sc_type = 1   -   Normalized spectral clustering according to Shi and Malik (2000)
            sc_type = 2   -   Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)
        """
        self.cluster_number = cluster_number

        self.D = np.zeros((self.point_number,self.point_number))
        for i in range(self.point_number):
            self.W[i,i] = 0
            self.D[i,i] = np.sum(self.W[i])
        self.L = self.D - self.W
        if sc_type == 1:
            D2 = np.diag(1. / self.D.diagonal())
            self.L = np.dot(D2,self.L)
        if sc_type == 2:
            D2 = np.diag(self.D.diagonal() ** (-.5))
            self.L = np.dot(D2,np.dot(self.L,D2))

        self.eig_val, self.eig_vect = la.eig(self.L)
        self._sortEig()
        self.points = self.eig_vect[:, 0:self.cluster_number]

        if sc_type == 2:
            for i in range(self.point_number):
                self.points[i] = self.points[i]/la.norm(self.points[i])

        if KMlib:
            codebook, cost = clust.kmeans(self.points, cluster_number, iter=KMiter)
            self.solution, cost2 = clust.vq(self.points, codebook)
        else:
            KM = KMean(self.points)
            self.KMdata = np.zeros(KMiter)
            bestCost = np.inf
            for i in range(KMiter):
                KM.run(cluster_number)
                sys.stdout.write(".")
                sys.stdout.flush()
                self.KMdata[i] = KM.cost
                #print i, ": ", KM.cost
                if bestCost > KM.cost:
                    bestCost = KM.cost
                    self.solution = KM.pointsNearestCluster
            sys.stdout.write("\n")
            sys.stdout.flush()

        return self.solution

    def evaluate_solution(self, solution):
        solution = self._swapSol(solution, self.solution, self.clusterNumber)
        return np.sum(solution == self.solution) * 1.0 / len(solution)


    def _kNearest(self, k):
        filter = np.zeros((self.point_number,self.point_number), dtype=bool)
        for i in range(self.point_number):
            knn = np.argsort(self.W[i])[-k:]
            filter[i,knn] = True
            filter[knn,i] = True
        self.W[filter==False] = 0

    def _kNearestMutual(self, k):
        k = int(k)
        filter = np.empty((self.point_number,self.point_number), dtype=bool)
        filter.fill(True)
        for i in range(self.point_number):
            x = heapq.nlargest(k, self.W[i])[k-1]
            filter[i] = filter[i] & (self.W[i]>=x)
            filter[:,i] = filter[:,i] & (self.W[i] >= x)

        self.W[filter == False] = 0

    def _sortEig(self):
        s = self.eig_val.argsort()
        self.eig_val = self.eig_val[s]
        self.eig_vect = self.eig_vect[:,s]

    def _swapSol(self, sol1, sol2):
        clusterNumber = len(set(sol2))
        best_pr = 0.0
        for p in all_perms(range(clusterNumber)):
            hits = 0.0
            for i in range(clusterNumber):
                hits += sum((sol1==i) & (sol2==p[i]))
            pr = hits/sol1.size
            if pr > best_pr:
                best_pr = pr
                best = p

        sol3 = np.copy(sol2)
        for i in range(clusterNumber):
            sol3[sol2 == best[i]] = i
        return sol3