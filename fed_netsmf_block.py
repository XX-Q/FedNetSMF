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

    parser.add_argument('--block_size', default=10, type=int,
                        help="block size of matices")
    parser.set_defaults(large=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp

    start_time = time.time()

    # Get parameters
    T = 0 # window size of random walk
    m = 0 # non-zeros of path sampling algorithm
    d = 0 # dimension of random svd





    # Initialize servers
    print("Initialize servers")
    maskingServer = MaskingServer()
    embeddingServer = EmbeddingServer()


    alice = DataHolder("alice")
    bob = DataHolder("bob")


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
    alice.path_sampling(alice.A_sparse, T, m, if_block=True, block_size=block_size)
    bob.path_sampling(bob.A_sparse, T, m, if_block=True, block_size=block_size)

    # Masking Server
    # Generate O
    maskingServer.generate_random_matrix(n, d)

    # Data Holders
    # Calculate M
    alice.calculate_M_block(D, maskingServer.O, block_size=block_size)
    bob.calculate_M_block(D, maskingServer.O, block_size=block_size)

    # Embedding Server
    # Calculate M sum
    embeddingServer.calculate_M_sum([alice.M_sparse, bob.M_sparse], block_size=block_size)
    # QR decomposition
    embeddingServer.calculate_qr_result(block_size=block_size)

    # Data Holders
    # Compute B
    alice.compute_B_block(embeddingServer.Q, block_size)
    bob.compute_B_block(embeddingServer.Q, block_size)

    # Output the result
    embedding_result = embeddingServer.calculate_svd_result(block_size)
























    # Get D-1A
    print("calculate D-1A")
    alice.get_DA_block(D)
    bob.get_DA_block(D)

    # Generate random matrices
    print("Generate random matrix O")
    maskingServer.generate_ramdom_matrix_O_block(block_size)





    alice.calculate_PDAQ_block(maskingServer.P,D,maskingServer.DQ)
    # alice.calculate_PDA2Q(maskingServer.P,D,maskingServer.DQ)
    bob.calculate_PDAQ_block(maskingServer.P, D, maskingServer.DQ)
    # bob.calculate_PDA2Q(maskingServer.P, D, maskingServer.DQ)

    # Start Calculate A1A2+A2A1
    # Sending random matrices to Alice
    print("Sending random matrices to Alice")
    alice.get_random_matrices(maskingServer.random_matrices_1)
    alice.get_T(maskingServer.T)

    # calculate matrix C
    print("calculate matrix C")
    maskingServer.calculate_C_block()

    # Sending random matrices to Bob
    print("Sending random matrices to Bob")
    bob.get_random_matrices(maskingServer.random_matrices_2)
    bob.get_C(maskingServer.C)



    # Bob send A_b - R to alice
    print("Bob send A_b - R to alice")
    alice.get_residual_2(bob.calculate_residual_2_block())

    bob.get_residual_1(alice.calculate_residual_1_block())
    bob.get_W(alice.calculate_W_block(size, block_size))

    Ts = alice.output_Ts_block()
    U = bob.output_U_block(block_size)
    # PDAAQs = np.dot(maskingServer.P, np.add(Ts,U)).dot(maskingServer.DQ)
    PDAAQs = output_PDAAQs(maskingServer.P, Ts, U, maskingServer.DQ)
    # finish calculate M
    factorizationServer.calculate_M_block(alice.PDAQ,bob.PDAQ,PDAAQs,vol, 2,args.negative, block_size)



    # Test Result
    M = factorizationServer.M
    Mshow = show_block_matrix(M, block_size)
    print("Masked fed output")
    print(Mshow)
    U_, s, V_ = svd(Mshow)
    # U = block_demask( block_matrix_splitter(U_,"U_",block_size),maskingServer.P,"U",block_size)
    # U = show_block_matrix(U,block_size)
    pt = show_block_matrix(maskingServer.P, block_size).T
    U = np.dot(pt, U_)
    res = np.diag(np.sqrt(s)).dot(U)
    print("Embedding result")
    print(res)
    np.save("evaluation/fed_deepwalk_embedding_1000.npy", res)

    end_time = time.time()
    print("total cost:", end_time-start_time)



    # U_, s, V_ = svd(M)
    # U = np.dot(maskingServer.P, U_)
    # res = np.diag(np.sqrt(s)).dot(U)
    # print("Embedding result")
    # print(res)






    # M = [np.array(m) for m in M]
    # print(M)





