import numpy as np
import torch
from .strategy import Strategy

class LeastConfidenceDropout(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10):
		super(LeastConfidenceDropout, self).__init__(X, Y, idxs_lb, net, handler, args)
		self.n_drop = n_drop

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob_dropout(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], self.n_drop)
		U = probs.max(1)[0]
		return idxs_unlabeled[U.sort()[1][:n]]
