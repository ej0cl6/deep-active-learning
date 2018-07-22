import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors

class KCenterGreedy(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(KCenterGreedy, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		lb_flag = self.idxs_lb.copy()
		embedding = self.get_embedding(self.X, self.Y)
		embedding = embedding.numpy()

		from datetime import datetime

		print('calculate distance matrix')
		t_start = datetime.now()
		dist_mat = np.matmul(embedding, embedding.transpose())
		sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
		dist_mat *= -2
		dist_mat += sq
		dist_mat += sq.transpose()
		dist_mat = np.sqrt(dist_mat)
		print(datetime.now() - t_start)

		mat = dist_mat[~lb_flag, :][:, lb_flag]

		for i in range(n):
			if i%10 == 0:
				print('greedy solution {}/{}'.format(i, n))
			mat_min = mat.min(axis=1)
			q_idx_ = mat_min.argmax()
			q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
			lb_flag[q_idx] = True
			mat = np.delete(mat, q_idx_, 0)
			mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

		return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]
