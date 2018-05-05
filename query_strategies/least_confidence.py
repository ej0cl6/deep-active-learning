import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
	def __init__(self, X, Y, idxs_lb, args):
		super(LeastConfidence, self).__init__(X, Y, idxs_lb, args)

	def query(self, n):
		idxs_unlabed = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabed], self.Y[idxs_unlabed])
		U = probs.max(1)[0]
		return idxs_unlabed[U.sort()[1][:n]]
