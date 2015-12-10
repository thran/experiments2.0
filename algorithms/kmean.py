import numpy as np
import random
import math
import pylab as plt
import scipy.stats as stats


class KMean:
	
	def __init__(self,data, metric = "euclid"): #row - one point; column - one feature
		self.points = data
		self.pointNumber = data.shape[0]
		self.featureNumber = data.shape[1]
		self.metric = metric
		
	
	def run(self,clusterNumber = 2, max_iterations = 50):
		self.clusterNumber = clusterNumber
		
		self.clustersInit()
		lastCost = 0
		for i in range(max_iterations):
			self.clustersMatch()
			self.clustersMove()
			self.cost = self.clusterCost()
			
			if lastCost == self.cost:
				break
			lastCost = self.cost
			
			self.iterationsDone = i+1
		
		return self.pointsNearestCluster
		
	def clustersInit(self):
		self.clusterPositions = np.empty((self.clusterNumber, self.featureNumber))
		self.pointsNearestCluster = np.empty((self.pointNumber),int)
		
		rand = range(self.pointNumber)
		random.shuffle(rand)
		
		for c in range(self.clusterNumber):
			self.clusterPositions[c,:] = self.points[rand[c],:]
			
			
	def clustersMatch(self):
		for p in range(self.pointNumber):
			bestDist = np.inf
			for c in range(self.clusterNumber):
				d = self.dist(self.points[p,:], self.clusterPositions[c,:])
				if d < bestDist:
					bestDist = d
					self.pointsNearestCluster[p] = c
					
	def clustersMove(self):
		for c in range(self.clusterNumber):
			m = self.points[(self.pointsNearestCluster == c),:]
			m = np.ma.masked_array(m, np.isnan(m))
			self.clusterPositions[c,:] = np.mean(m,0)
			
	def clusterCost(self):
		c = 0
		for p in range(self.pointNumber):
			c += self.dist(self.points[p,:], self.clusterPositions[self.pointsNearestCluster[p]])
		return c/self.pointNumber
		
		
	def dist(self,x,y):
		if self.metric=="euclid":
			if np.size(x) == np.size(y):
				d = 0
				for i in range(np.size(x)):
					if not np.isnan(x[i]) and not np.isnan(y[i]):
						d += abs(y[i]-x[i])**2
				return math.sqrt(d)
		
		if self.metric=="spearman":
			filt = (np.isnan(x) + np.isnan(y)) == 0
			x = x[filt]
			y = y[filt]
			
			d = stats.spearmanr(x,y)[0]
			
			if np.isnan(d): return 0
			
			return d
			
		if self.metric=="pearson":
			filt = (np.isnan(x) + np.isnan(y)) == 0
			x = x[filt]
			y = y[filt]
			
			d = stats.pearsonr(x,y)[0]
			
			if np.isnan(d): return 0
			
			return d
			
			
def demo():
	dat = 1000
	a = np.empty((3*dat,2),float)
	for i in range(dat):
		a[i] = np.random.multivariate_normal([0,0],[[1,0.1],[1,.7]])
		a[i+dat] = np.random.multivariate_normal([3,2],[[0.5,0],[0,2]])
		a[i+2*dat] = np.random.multivariate_normal([3,2],[[0.5,0],[0,2]])
		#a[i+3*dat] = np.random.multivariate_normal([0,4],[[0.1,0],[0,.1]])
	
	
	km = KMean(a)
	km.run(2)
	#z = (km.pointsNearestCluster == 2)
	x = (km.pointsNearestCluster == 1)
	y = (km.pointsNearestCluster == 0)
	
	
	a[np.isnan(a)] = -10
	
	
	plt.plot(a[x,0],a[x,1],"ro")
	plt.plot(a[y,0],a[y,1],"bo")
	#plt.plot(a[z,0],a[z,1],"go")
	plt.show()
	
#demo()
