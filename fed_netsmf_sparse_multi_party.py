# encoding: utf-8

import scipy.io
import scipy.sparse as sp
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging
import theano
from theano import tensor as T
from sklearn import preprocessing
from sklearn.utils.extmath import *
from data_utils import *
from entity.DataHolder import DataHolder
from entity.EmbeddingServer import EmbeddingServer
from entity.MaskingServer import MaskingServer
from utils import *
import time
from multiprocessing import Pool
from clf import *
import sys

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

    dataset_name = "ppi"
    total_communication = 0
    # Get parameters
    m_ = 500  # round parameter of FedNetSMF
    T = 1  # window size of random walk
    num_parties = 2  # number of federated parties
    d = 128  # embedding dimension
    b = 1  # negative sampling
    num_workers = 8
    num_iter = 7
    is_directed = True

    parties = []
    parties_data = []
    # parties_data = ["datasets/blogcatalog/edges_0_3_10312_333983.txt",
    #                 "datasets/blogcatalog/edges_1_3_10312_333983.txt",
    #                 "datasets/blogcatalog/edges_2_3_10312_333983.txt"
    #                 ]
    for i in range(num_parties):
        parties_data.append(f"datasets/{dataset_name}/edges_{str(i)}_{num_parties}_3890_76584.txt")
        # _80513_11799764
        # _10312_667966
    # parties_data = ["datasets/blogcatalog/edges_0_5_10312_333983.txt",
    #                 "datasets/blogcatalog/edges_1_5_10312_333983.txt",
    #                 "datasets/blogcatalog/edges_2_5_10312_333983.txt",
    #                 "datasets/blogcatalog/edges_3_5_10312_333983.txt",
    #                 "datasets/blogcatalog/edges_4_5_10312_333983.txt"
    #                 ]
    # parties_data = ["datasets/blogcatalog/edges_0_2_10312_333983.txt","datasets/blogcatalog/edges_1_2_10312_333983.txt"]

    # Initialize servers
    print("Initialize servers")
    maskingServer = MaskingServer()
    embeddingServer = EmbeddingServer()

    for i in range(num_parties):
        parties.append(DataHolder(f"DH{i}"))

    # alice = DataHolder("alice")
    # bob = DataHolder("bob")

    # load data
    # TODO rewrite after test

    print("initalize dataholder")


    time1 = time.time()

    print(parties_data)
    for i in range(num_parties):
        parties[i].load_data_edgelist_sparse(parties_data[i], is_directed=is_directed)

    m = 0  # total edges of original network
    D = []
    vol = 0
    for i in range(num_parties):
        if i == 0:
            m = parties[i].vol
            D = parties[i].D
            vol = parties[i].vol

            total_communication+=sys.getsizeof(m)
            total_communication += sys.getsizeof(D)
            total_communication += sys.getsizeof(vol)
        else:
            m += parties[i].vol
            D += parties[i].D
            vol += parties[i].vol

            total_communication += sys.getsizeof(m)
            total_communication += sys.getsizeof(D)
            total_communication += sys.getsizeof(vol)
    D_inv = D.power(-1)
    n = parties[0].total_node

    print(m)
    print(D)
    print(vol)
    # Data Holders
    # Path sampling
    for i in range(num_parties):
        tmp_result = []
        pool = Pool(processes=num_workers)
        for j in range(num_workers):
            tmp_result.append(
                pool.apply_async(func=parties[i].random_walk_sparse, args=(i, m_, T, num_workers, num_parties)))
        pool.close()
        pool.join()
        tmp_A = sp.csr_matrix((n, n))
        for res in tmp_result:
            tmp_A += res.get()
        parties[i].calculate_L_sparse(tmp_A)


    print(1)

    maskingServer.generate_random_matrix_sparse(n, d)

    for i in range(num_parties):
        parties[i].calculate_M_sparse_base(D_inv, vol=vol, b=b)

        total_communication += sys.getsizeof(D_inv)



    # QR by iterate
    for i in range(num_iter + 1):
        if i == 0:
            for j in range(num_parties):
                parties[j].calculate_M_sparse_iterate(parties[j].M_sparse, maskingServer.O)

                total_communication += sys.getsizeof(maskingServer.O)
                total_communication += sys.getsizeof(parties[j].M_iterate)

            embeddingServer.calculate_M_sum_sparse([p.M_iterate for p in parties])

            for j in range(num_parties):
                parties[j].calculate_M_sparse_iterate(parties[j].M_sparse.T, embeddingServer.M_sum)

                total_communication += sys.getsizeof(embeddingServer.M_sum)
                total_communication += sys.getsizeof(parties[j].M_iterate)
            embeddingServer.calculate_M_sum_sparse([p.M_iterate for p in parties])
        elif i == num_iter:
            for j in range(num_parties):
                parties[j].calculate_M_sparse_iterate(parties[j].M_sparse, embeddingServer.M_sum)

                total_communication += sys.getsizeof(embeddingServer.M_sum)
                total_communication += sys.getsizeof(parties[j].M_iterate)
            embeddingServer.calculate_M_sum_sparse([p.M_iterate for p in parties])

        else:
            for j in range(num_parties):
                parties[j].calculate_M_sparse_iterate(parties[j].M_sparse, embeddingServer.M_sum)

                total_communication += sys.getsizeof(embeddingServer.M_sum)
                total_communication += sys.getsizeof(parties[j].M_iterate)
            embeddingServer.calculate_M_sum_sparse([p.M_iterate for p in parties])

            for j in range(num_parties):
                parties[j].calculate_M_sparse_iterate(parties[j].M_sparse.T, embeddingServer.M_sum)

                total_communication += sys.getsizeof(embeddingServer.M_sum)
                total_communication += sys.getsizeof(parties[j].M_iterate)
            embeddingServer.calculate_M_sum_sparse([p.M_iterate for p in parties])

    # QR decomposition
    embeddingServer.calculate_qr_result_sparse()

    # Data Holders
    # Compute B
    for i in range(num_parties):
        parties[i].compute_B_sparse(embeddingServer.Q)

        total_communication += sys.getsizeof(embeddingServer.Q)
        total_communication += sys.getsizeof(parties[i].B)

    # Compute B sum
    embeddingServer.calculate_B_sum_sparse([p.B for p in parties])

    # Output the result
    embedding_result = embeddingServer.calculate_svd_result_sparse()


    def _get_embedding_rand(matrix):
        t1 = time.time()
        l = matrix.shape[0]  # noqa E741
        smat = sp.csc_matrix(matrix)
        print("svd sparse", smat.data.shape[0] * 1.0 / l ** 2)
        U, Sigma, VT = randomized_svd(smat, n_components=128, n_iter=5, random_state=None)
        U = U * np.sqrt(Sigma)
        U = preprocessing.normalize(U, "l2")
        print("sparsesvd time", time.time() - t1)
        return U


    print(embedding_result.shape)
    if num_parties == 2:
        np.save(f"embedding/fednetsmf_{dataset_name}.npy", embedding_result)
    else:
        np.save(f"embedding/fednetsmf_{dataset_name}_{num_parties}.npy", embedding_result)

    time2 = time.time()
    evaluate(time2 - time1, "fednetsmf", dataset_name, num_parties, total_communication)

    print(embedding_result)


