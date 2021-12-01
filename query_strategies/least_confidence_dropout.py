import numpy as np
from .strategy import Strategy

class LeastConfidenceDropout(Strategy):
    def __init__(self, dataset, net, n_drop=10):
        super(LeastConfidenceDropout, self).__init__(dataset, net)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob_dropout(unlabeled_data, n_drop=self.n_drop)
        uncertainties = probs.max(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
