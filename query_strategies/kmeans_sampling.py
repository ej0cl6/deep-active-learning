import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, args):
		super(KMeansSampling, self).__init__(X, Y, idxs_lb, args)

	def query(self, n):
		idxs_unlabed = np.arange(self.n_pool)[~self.idxs_lb]
		embedding = self.get_embedding(self.X[idxs_unlabed], self.Y[idxs_unlabed])
		embedding = embedding.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(embedding)
		distances = cluster_learner.transform(embedding)
		return idxs_unlabed[distances.argmin(axis=0)]
