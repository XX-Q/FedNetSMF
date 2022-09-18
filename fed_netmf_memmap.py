#!/usr/bin/env python
# encoding: utf-8
# File Name: eigen.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/13 16:05
# TODO:

import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
from theano import tensor as T

from entity.DataHolder import DataHolder
from entity.FactorizationServer import FactorizationServer
from entity.MaskingServer import MaskingServer
from evalueate import evaluate_MAE
from utils import *

logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='datasets/blogcatalog.mat',
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
    parser.set_defaults(large=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp

    # Initialize servers
    maskingServer = MaskingServer()
    factorizationServer = FactorizationServer()

    # 两个dataholder分别命名为alice和bob
    alice = DataHolder("alice")
    bob = DataHolder("bob")

    # load data
    # TODO rewrite after test
    # alice.load_data(args.input)
    # bob.load_data(args.input)

    # load test data
    # size = 10
    alice.load_test_data_1000(1000)
    bob.load_test_data_1000(1000)

    # Calculate D
    # D = alice.D+bob.D
    # TODO delete this
    A = alice.A_own + bob.A_own
    vol = alice.vol = bob.vol = float(A.sum())
    # L, D = csgraph.laplacian(A, normed=True, return_diag=True)
    D = alice.D+bob.D
    # Get D-1A
    alice.get_DA(D)
    bob.get_DA(D)

    # Generate orthogonal matrices
    maskingServer.size = D.shape[0]
    maskingServer.generate_orthogonal_matrices(D)
    # Generate random matrices
    maskingServer.generate_random_matrices()
    maskingServer.generate_random_matrix_T()

    alice.calculate_PDAQ(maskingServer.P,D,maskingServer.DQ)
    alice.calculate_PDA2Q(maskingServer.P,D,maskingServer.DQ)
    bob.calculate_PDAQ(maskingServer.P, D, maskingServer.DQ)
    bob.calculate_PDA2Q(maskingServer.P, D, maskingServer.DQ)

    # Start Calculate A1A2+A2A1
    # Sending random matrices to Alice
    alice.get_random_matrices(maskingServer.random_matrices_1)
    alice.get_T(maskingServer.T)

    # calculate matrix C
    maskingServer.calculate_C()

    # Sending random matrices to Bob
    bob.get_random_matrices(maskingServer.random_matrices_2)
    bob.get_C(maskingServer.C)



    # Bob send A_b - R to alice
    alice.get_residual_2(bob.calculate_residual_2())

    bob.get_residual_1(alice.calculate_residual_1())
    bob.get_W(alice.calculate_W())

    Ts = alice.output_Ts()
    U = bob.output_U()
    PDAAQs = np.dot(maskingServer.P, np.add(Ts,U)).dot(maskingServer.DQ)
    # finish calculate M
    factorizationServer.calculate_M(alice.PDAQ,bob.PDAQ,PDAAQs,vol, 2,args.negative)




    # Test Result
    M = factorizationServer.M
    print("Masked fed output")
    print(M)


    U_, s, V_ = svd(M)
    U = np.dot(maskingServer.P.T, U_)
    res = np.diag(np.sqrt(s)).dot(U)
    print("Embedding result")
    print(res)


    # Mtest = test_line(A, D, args.negative, maskingServer.P, maskingServer.Q)
    Mtest = test_netmf(A,D,2,args.negative,maskingServer.P,maskingServer.Q)
    U_, s, V_ = svd(Mtest)
    U = np.dot(maskingServer.P.T, U_)
    rest = np.diag(np.sqrt(s)).dot(U)
    print(rest)

    print("LINE MAE:", evaluate_MAE(A, rest))

    np.save("evaluation/deepwalk_embedding_1000.npy", rest)






    # M = [np.array(m) for m in M]
    # print(M)





