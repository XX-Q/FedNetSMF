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
from entity.FactorizationServer import FactorizationServer
from entity.MaskingServer import MaskingServer
from utils import *
import time
from evalueate import *
import scipy.sparse as sp
from clf import *



logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='datasets/soc-Epinions1.mat',
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
                        format='%(asctime)s %(message)s')  # include timestamp2108

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
    # bob.A_own = alice.A_own
    # bob.D = alice.D
    # bob.vol = alice.vol
    #bob.load_data_block(args.input,block_size)

    # load test data
    # size = 75888
    # size = 1000
    # # print("initalize dataholder")
    # A1 = alice.load_test_data_block_1000(size, block_size)
    # A2 = bob.load_test_data_block_1000(size, block_size)

    size = 10312
    alice.load_data_block_new(size, block_size,"datasets/blogcatalog/edges_0_2_10312_667966.txt")
    bob.load_data_block_new(size, block_size,"datasets/blogcatalog/edges_1_2_10312_667966.txt")
    #bob.A_own = alice.A_own
    #bob.D = alice.D
    #bob.vol = alice.vol


    # Calculate D
    D = alice.D+bob.D

    vol = alice.vol + bob.vol

    D = block_matrix_splitter(D, "D", block_size, diag=True)

    # del varible
    del alice.D, bob.D
    gc.collect()

    # Get D-1A
    #print("calculate D-1A")
    logger.info("calculate D-1A")
    alice.get_DA_block(D)
    bob.get_DA_block(D)

    # Generate orthogonal matrices
    logger.info("Generate orthogonal matrices")
    maskingServer.size = size
    p, q = maskingServer.generate_orthogonal_matrices_block(D, block_size)
    # Generate random matrices
    # logger.info("Generate random matrices")
    # maskingServer.generate_random_matrices_block(block_size)
    # maskingServer.generate_ramdom_matrix_T_block(block_size)

    logger.info("calculate PDAQ")
    alice.calculate_PDAQ_block(maskingServer.P,D,maskingServer.DQ)
    # alice.calculate_PDA2Q(maskingServer.P,D,maskingServer.DQ)
    bob.calculate_PDAQ_block(maskingServer.P, D, maskingServer.DQ)
    # bob.calculate_PDA2Q(maskingServer.P, D, maskingServer.DQ)
    gc.collect()

    # Start Calculate A1A2+A2A1
    # Sending random matrices to Alice
    # logger.info("Sending random matrices to Alice")
    # alice.get_random_matrices(maskingServer.random_matrices_1)
    # alice.get_T(maskingServer.T)

    # calculate matrix C
    # logger.info("calculate matrix C")
    # maskingServer.calculate_C_block()
    # logger.info("calculate matrix C finished")
    # gc.collect()

    # Sending random matrices to Bob
    # logger.info("Sending random matrices to Bob")
    # bob.get_random_matrices(maskingServer.random_matrices_2)
    # bob.get_C(maskingServer.C)



    # Bob send A_b - R to alice
    # logger.info("Bob send A_b - R to alice")
    # alice.get_residual_2(bob.calculate_residual_2_block())

    # bob.get_residual_1(alice.calculate_residual_1_block())
    # logger.info("calculate W")
    # bob.get_W(alice.calculate_W_block(size, block_size))
    # logger.info("calculate W finished")
    # gc.collect()

    # Ts = alice.output_Ts_block()
    # logger.info("calculate U")
    # U = bob.output_U_block(block_size)
    # logger.info("calculate U finished")
    # gc.collect()
    # PDAAQs = np.dot(maskingServer.P, np.add(Ts,U)).dot(maskingServer.DQ)
    # logger.info("calculate PDAAQs")
    # PDAAQs = output_PDAAQs(maskingServer.P, Ts, U, maskingServer.DQ)
    # logger.info("calculate PDAAQs finished")
    # finish calculate M
    logger.info("calculate M")
    factorizationServer.calculate_M_line_block(alice.PDAQ,bob.PDAQ,vol, args.negative, block_size)
    logger.info("calculate M finished")
    gc.collect()



    # Test Result
    M = factorizationServer.M
    Mshow = show_block_matrix(M, block_size, shape=(size, size))
    Mshow = Mshow[:size, :size]
    print("Masked fed output")
    print(Mshow)
    qt = show_block_matrix(maskingServer.Q, block_size, (size, size)).T[:size, :size]
    pt = show_block_matrix(maskingServer.P, block_size, (size, size)).T[:size, :size]

    Mshow = np.dot(pt, Mshow).dot(qt)

    Mshow[Mshow <= 1] = 1
    Mshow = np.log(Mshow)
    print("fed output before svd")
    print(Mshow)

    u, s, _ = sp.linalg.svds(Mshow, 128)
    # U_, s, V_ = svd(Mshow)
    # U = block_demask( block_matrix_splitter(U_,"U_",block_size),maskingServer.P,"U",block_size)
    # U = show_block_matrix(U,block_size)

    # U = np.dot(pt, U_)
    # U[U <= 1] = 1
    # U = np.log(U)
    # res = np.diag(np.sqrt(s)).dot(U_.T).T
    res = sp.diags(np.sqrt(s)).dot(u.T).T
    # res = U_.dot(np.diag(np.sqrt(s)))
    # res = res[:size, :size]
    np.save("embedding/fedline_blogcatalog.npy",res)

    end_time = time.time()
    print("total cost:", end_time-start_time)

    evaluate(end_time - start_time, "fedline", "blogcatalog")








    # M = [np.array(m) for m in M]
    # print(M)





