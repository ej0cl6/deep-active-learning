import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy

class AdversarialDeepFool(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args, max_iter=50):
        super(AdversarialDeepFool, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.max_iter = max_iter

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out, e1 = self.clf(nx+eta)
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

            eta += ri.clone()
            nx.grad.data.zero_()
            out, e1 = self.clf(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        self.clf.cpu()
        self.clf.eval()
        dis = np.zeros(idxs_unlabeled.shape)

        data_pool = self.handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], transform=self.args['transform'])
        for i in range(len(idxs_unlabeled)):
            if i % 100 == 0:
                print('adv {}/{}'.format(i, len(idxs_unlabeled)))
            x, y, idx = data_pool[i]
            dis[i] = self.cal_dis(x)

        self.clf.cuda()

        return idxs_unlabeled[dis.argsort()[:n]]


