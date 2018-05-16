import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors

class KCenterGreedy(Strategy):
	def __init__(self, X, Y, idxs_lb, args):
		super(KCenterGreedy, self).__init__(X, Y, idxs_lb, args)

	def query(self, n):
		idxs_lb_copy = self.idxs_lb.copy()
		for i in range(n):
			print('n_center {}/{}'.format(i+1, n))
			idxs_lb = np.arange(self.n_pool)[idxs_lb_copy]
			idxs_ub = np.arange(self.n_pool)[~idxs_lb_copy]
			embedding = self.get_embedding(self.X, self.Y)
			embedding = embedding.numpy()
			embedding_lb = embedding[idxs_lb]
			embedding_ub = embedding[idxs_ub]

			NN = NearestNeighbors(n_neighbors=1)
			NN.fit(embedding_lb)
			nn_dis, nn_idxs = NN.kneighbors(embedding_ub)
			q_idx = idxs_ub[nn_dis.argmax()]
			idxs_lb_copy[q_idx] = True

		return np.arange(self.n_pool)[(self.idxs_lb ^ idxs_lb_copy)]
