import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

class AdversarialDeepFool(Strategy):
    def __init__(self, dataset, net, max_iter=50):
        super(AdversarialDeepFool, self).__init__(dataset, net)
        self.max_iter = max_iter

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out, e1 = self.net.clf(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi
                    value_l = value_i

            eta += ri.clone()
            nx.grad.data.zero_()
            out, e1 = self.net.clf(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        self.net.clf.cpu()
        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)

        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            dis[i] = self.cal_dis(x)

        self.net.clf.cuda()

        return unlabeled_idxs[dis.argsort()[:n]]


