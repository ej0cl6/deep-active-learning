import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
    def __init__(self, dataset, net):
        super(KMeansSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)
        
        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

        return unlabeled_idxs[q_idxs]
