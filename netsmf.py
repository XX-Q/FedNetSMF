import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
import gc

from entity.DataHolder import DataHolder
from entity.EmbeddingServer import EmbeddingServer
from entity.MaskingServer import MaskingServer
from utils import *
import time
logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'

# Get parameters
T = 4  # window size of random walk
m = 100  # non-zeros of path sampling algorithm
d = 50  # dimension of random svd

A = np.load("test_A_100.npy")


A_sparse = sparse.dok_matrix(A)
A = A_sparse.todense()
D = np.array(sum(A), dtype=float)


# path sampling
A_sample_sprase = path_sampling(A_sparse, T, m)

A_sample = A_sample_sprase.todense()

# np.save("test_A_100_sampled.npy", A_sample)

A_sample = np.load("test_A_100_sampled.npy")

# A_sample = A

D_sample = np.array(sum(A_sample), dtype=float)

# generate random matrix
O = random_matrix_unblock(len(A_sample), d)



u, s, v = rsvd(A_sample, O)

res = np.dot(u, np.diag(np.sqrt(s)))

res = np.array(res)

print(res)