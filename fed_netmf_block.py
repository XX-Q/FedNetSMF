# encoding: utf-8

import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
import scipy.sparse as sp
from clf import *
from theano import tensor as T
import gc

from entity.DataHolder import DataHolder
from entity.FactorizationServer import FactorizationServer
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

    parser.add_argument('--block_size', default=389, type=int,
                        help="block size of matices")
    parser.set_defaults(large=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp

    start_time = time.time()
    # Initialize servers
    print("Initialize servers")
    maskingServer = MaskingServer()
    factorizationServer = FactorizationServer()


    alice = DataHolder("alice")
    bob = DataHolder("bob")


    block_size = args.block_size

    # load data
    # TODO rewrite after test
    print("initalize dataholder")
    # alice.load_data_block(args.input,block_size)
    # bob.load_data_block(args.input,block_size)

    # load test data
    # print("initalize dataholder")
    # size = 1000
    # alice.load_test_sparse_data_block_1000(size, block_size,0.001)
    # bob.load_test_sparse_data_block_1000(size, block_size,0.001)

    size = 3890
    alice.load_data_block_new(size, block_size,"datasets/ppi/edges_0_2_3890_76584.txt")
    bob.load_data_block_new(size, block_size,"datasets/ppi/edges_1_2_3890_76584.txt")


    # Calculate D
    D = alice.D+bob.D

    vol = alice.vol + bob.vol

    D = block_matrix_splitter(D, "D", block_size, diag=True)

    # del varible
    del alice.D, bob.D
    gc.collect()

    # Get D-1A
    print("calculate D-1A")
    alice.get_DA_block(D)
    bob.get_DA_block(D)

    # Generate orthogonal matrices
    print("Generate orthogonal matrices")
    maskingServer.size = size
    maskingServer.generate_orthogonal_matrices_block(D, block_size)
    # Generate random matrices
    print("Generate random matrices")
    maskingServer.generate_random_matrices_block(block_size)
    maskingServer.generate_ramdom_matrix_T_block(block_size)

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
    U = bob.output_U_block(block_size, size)
    # PDAAQs = np.dot(maskingServer.P, np.add(Ts,U)).dot(maskingServer.DQ)
    PDAAQs = output_PDAAQs(maskingServer.P, Ts, U, maskingServer.DQ)
    # finish calculate M
    factorizationServer.calculate_M_block(alice.PDAQ,bob.PDAQ,PDAAQs,vol, 2,args.negative, block_size)



    # Test Result
    M = factorizationServer.M
    Mshow = show_block_matrix(M, block_size, shape=(size,size))
    Mshow = Mshow[:size, :size]
    print("Masked fed output")
    print(Mshow)
    qt = show_block_matrix(maskingServer.Q, block_size,(size,size)).T[:size,:size]
    pt = show_block_matrix(maskingServer.P, block_size,(size,size)).T[:size,:size]


    Mshow = np.dot(pt, Mshow).dot(qt)

    Mshow[Mshow <= 1] = 1
    Mshow = np.log(Mshow)
    print("fed output before svd")
    print(Mshow)

    u, s, _ = sp.linalg.svds(Mshow, 128)
    res = sp.diags(np.sqrt(s)).dot(u.T).T

    print("Embedding result")
    print(res)
    np.save("embedding/fednetmf_ppi.npy", res)
    # np.save("evaluation/ppi_1000.npy", res[:, :128])
    end_time = time.time()

    print("total cost:", end_time-start_time)

    evaluate(end_time-start_time, "fednetmf", "ppi")







