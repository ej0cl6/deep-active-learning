import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(KMeansSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		embedding = self.get_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		embedding = embedding.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(embedding)
		
		cluster_idxs = cluster_learner.predict(embedding)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (embedding - centers)**2
		dis = dis.sum(axis=1)
		q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

		return idxs_unlabeled[q_idxs]
