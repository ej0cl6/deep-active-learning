import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, KMeansSampling, KCenterGreedy, BALDDropout

import ipdb

# parameters
SEED = 5

NUM_INIT_LB = 100
NUM_QUERY = 100
NUM_ROUND = 10

NUM_EPOCH = 25
BATCH_SIZE_TR = 10
BATCH_SIZE_TE = 1000
NUM_WORKER = 1
LEARNING_RATE = 0.01
MOMENTUM = 0.5

TRANSFORM = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

args = {'n_epoch': NUM_EPOCH, 'transform': TRANSFORM,
        'loader_tr_args':{'batch_size': BATCH_SIZE_TR, 'num_workers': NUM_WORKER},
        'loader_te_args':{'batch_size': BATCH_SIZE_TE, 'num_workers': NUM_WORKER},
        'optimizer_args':{'lr': LEARNING_RATE, 'momentum': MOMENTUM}}

# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# load dataset
raw_tr = datasets.MNIST('./MNIST', train=True, download=True)
raw_te = datasets.MNIST('./MNIST', train=False, download=True)
# raw_tr = datasets.FashionMNIST('./FashionMNIST', train=True, download=True)
# raw_te = datasets.FashionMNIST('./FashionMNIST', train=False, download=True)
X_tr = raw_tr.train_data
Y_tr = raw_tr.train_labels
X_te = raw_te.test_data
Y_te = raw_te.test_labels

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))

# generate initial label
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_lb[np.random.randint(0, n_pool, NUM_INIT_LB)] = True

# round 0 accuracy
# strategy = RandomSampling(X_tr, Y_tr, idxs_lb, args)
# strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, args)
# strategy = MarginSampling(X_tr, Y_tr, idxs_lb, args)
# strategy = EntropySampling(X_tr, Y_tr, idxs_lb, args)
# strategy = LeastConfidenceDropout(X_tr, Y_tr, idxs_lb, args, n_drop=100)
# strategy = MarginSamplingDropout(X_tr, Y_tr, idxs_lb, args, n_drop=100)
# strategy = EntropySamplingDropout(X_tr, Y_tr, idxs_lb, args, n_drop=100)
# strategy = KMeansSampling(X_tr, Y_tr, idxs_lb, args)
# strategy = KCenterGreedy(X_tr, Y_tr, idxs_lb, args)
strategy = BALDDropout(X_tr, Y_tr, idxs_lb, args, n_drop=100)

strategy.train()
print(type(strategy).__name__)
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
print('Round 0\ntesting accuracy {}'.format(acc[0]))

for rd in range(1, NUM_ROUND+1):
    print('Round {}'.format(rd))

    q_idxs = strategy.query(NUM_QUERY)
    idxs_lb[q_idxs] = True
    strategy.update(idxs_lb)
    strategy.train()
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
    print('testing accuracy {}'.format(acc[rd]))

print(type(strategy).__name__)
print(acc)
