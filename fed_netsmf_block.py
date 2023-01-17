# encoding: utf-8

import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
from theano import tensor as T
import gc

from entity.DataHolder import DataHolder
from entity.EmbeddingServer import EmbeddingServer
from entity.MaskingServer import MaskingServer
from utils import *
import time
logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='datasets/Homo_sapiens.mat',
                        help=".mat input file path")
    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--output", type=str, default='out/',
                        help="embedding output file path")

    parser.add_argument("--rank", default=256, type=int,
                        help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
                        help="dimension of embedding")
    parser.add_argument("--window", default=4,
                        type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
                        help="negative sampling")

    parser.add_argument('--large', dest="large", action="store_true",
                        help="using netmf for large window size")
    parser.add_argument('--small', dest="large", action="store_false",
                        help="using netmf for small window size")

    parser.add_argument('--block_size', default=100, type=int,
                        help="block size of matices")
    parser.set_defaults(large=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp

    start_time = time.time()

    # Get parameters
    T = 3 # window size of random walk
    m_ = 100000 # non-zeros of path sampling algorithm
    d = 100 # dimension of random svd





    # Initialize servers
    print("Initialize servers")
    maskingServer = MaskingServer()
    embeddingServer = EmbeddingServer()


    alice = DataHolder("alice")
    bob = DataHolder("bob")

    ma = 0
    mb = 0 # edges in original network


    block_size = args.block_size

    # load data
    # TODO rewrite after test
    print("initalize dataholder")

    # load test data
    size = 1000
    # print("initalize dataholder")
    alice.load_test_data_block_1000(size, block_size)
    bob.load_test_data_block_1000(size, block_size)

    # Calculate D
    # D = secure_aggregation(alice.D+bob.D)
    D = alice.D+bob.D
    n = len(D)

    vol = alice.vol + bob.vol

    D = block_matrix_splitter(D, "D", block_size, diag=True)

    # del varible
    # del alice.D, bob.D
    # gc.collect()

    # Data Holders
    # Path sampling
    alice.path_sampling_test(alice.A_sparse, T, m, if_block=True, block_size=block_size)
    bob.path_sampling_test(bob.A_sparse, T, m, if_block=True, block_size=block_size)

    # Masking Server
    # Generate O
    maskingServer.generate_random_matrix_test(n, d, block_size=block_size)

    # Data Holders
    # Calculate M
    D_ = show_block_matrix(D, block_size)
    D_ = np.diag(D_ ** -1)
    D_ = block_matrix_splitter(D_, name="D_", diag=True, block_size=block_size)
    alice.calculate_M_block(D_, maskingServer.O, block_size=block_size)
    bob.calculate_M_block(D_, maskingServer.O, block_size=block_size)

    # Embedding Server
    # Calculate M sum
    embeddingServer.calculate_M_sum([alice.M_sparse, bob.M_sparse], block_size=block_size)
    # QR decomposition
    embeddingServer.calculate_qr_result(block_size=block_size)

    # Data Holders
    # Compute B
    alice.compute_B_block(embeddingServer.Q, block_size)
    bob.compute_B_block(embeddingServer.Q, block_size)

    embeddingServer.calculate_B_sum([alice.B, bob.B], block_size=block_size)
    # Output the result
    # TODO 检查一下结果的形状
    # TODO 固定变量检查结果
    embedding_result = embeddingServer.calculate_svd_result(block_size)

    print(embedding_result)


