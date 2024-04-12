import numpy as np

from utils import *
import scipy.io
import scipy.sparse as sp
from scipy.sparse import *
from scipy.sparse import csgraph
import logging
import theano
from tqdm import tqdm
from sklearn.utils.extmath import *
import networkx as nx


logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'


class DataHolder():
    def __init__(self, name):
        super(DataHolder, self).__init__()
        self.name = name

        self.DA_own = []
        self.A_own = []
        self.A_sparse = []
        self.A_other = []


        self.vol = 0.0
        self.num_node = 0
        self.total_node = 0
        self.edgelist = []
        self.neighbors = {}

        self.D = []
        self.L = []

        self.D_sparse = []
        self.L_sparse = []

        self.M_sparse = []

        self.PDAQ = []
        self.PDA2Q = []

        # Alice
        self.random_matrices = []
        self.T = []
        self.T_ = []
        self.residual_1 = []
        self.residual_2 = []

        # Bob
        self.C = []
        self.W = []

    def generate_random_adjacency_matrix(self, size, if_sparse=False):
        res = np.random.randint(2, size=(size, size))
        res = np.triu(res)
        res = res + res.T - np.diag(res.diagonal())
        if if_sparse:
            return sp.csc_matrix(res)
        return res

    def load_test_data(self, size):
        '''
        load test matrix to test the code
        :param size:size of initalize
        :return:
        '''
        self.A_own = self.generate_random_adjacency_matrix(size, if_sparse=False)
        self.vol = float(self.A_own.sum())
        # self.L, self.D = csgraph.laplacian(self.A_own, normed=True, return_diag=True)
        self.D = np.array(sum(self.A_own), dtype=float)
    def load_test_data_1000(self, size):
        self.A_own = np.load("datasets/test_"+self.name+"_1000.npy")
        self.vol = float(self.A_own.sum())
        # self.L, self.D = csgraph.laplacian(self.A_own, normed=True, return_diag=True)
        self.D = np.array(sum(self.A_own), dtype=float)

    def load_test_data_block(self, size, block_size):
        res = self.A_own = self.generate_random_adjacency_matrix(size, if_sparse=False)
        self.vol = float(self.A_own.sum())
        # self.L, self.D = csgraph.laplacian(self.A_own, normed=True, return_diag=True)
        self.D = np.array(sum(self.A_own), dtype=float)
        self.A_sparse = dok_matrix(self.A_own)
        self.A_own = block_matrix_splitter(self.A_own, name=self.name, diag=False, block_size=block_size)
        return res

    def load_test_data_block_1000(self, size, block_size):
        self.A_own = np.load("datasets/test_"+self.name+"_1000.npy")
        self.vol = float(self.A_own.sum())
        # self.L, self.D = csgraph.laplacian(self.A_own, normed=True, return_diag=True)
        self.D = np.array(sum(self.A_own), dtype=float)
        self.A_sparse = dok_matrix(self.A_own)
        self.A_own = block_matrix_splitter(self.A_own, name=self.name, diag=False, block_size=block_size)

    def load_test_sparse_data_block_1000(self, size, block_size, sparse_rate):
        self.A_own = np.load("datasets/test_"+self.name+"_"+str(sparse_rate)+"_1000.npy")
        self.vol = float(self.A_own.sum())
        # self.L, self.D = csgraph.laplacian(self.A_own, normed=True, return_diag=True)
        self.D = np.array(sum(self.A_own), dtype=float)
        self.A_sparse = dok_matrix(self.A_own)
        self.A_own = block_matrix_splitter(self.A_own, name=self.name, diag=False, block_size=block_size)


    def load_data_block_new(self, size, block_size, path):
        info_list = path[:-4].split("_")[1:]
        print(info_list)
        num_part = int(info_list[0])
        total_part = int(info_list[1])
        self.total_node = int(info_list[2])
        total_edge = int(info_list[3])
        self.A_sparse = dok_matrix((self.total_node, self.total_node))

        with open(path, "r") as fr:
            self.edgelist = [[int(edges.split(" ")[0]), int(edges.split(" ")[1])] for edges in fr.readlines()]
        for edges in self.edgelist:
            # m, n = edges.split(" ")
            m, n = edges
            self.A_sparse[m, n] = 1

        # self.neighbors1 = []
        # A_paths = list(dict(self.A_sparse).keys())
        # for path in A_paths:
        #     if path[0] not in list(self.neighbors1):
        #         self.neighbors1[path[0]] = [path[1]]
        #     self.neighbors1[path[0]].append(path[1])

        self.A_own = self.A_sparse.todense()
        # self.A_own = np.load(path)
        self.vol = float(self.A_own.sum())
        # self.L, self.D = csgraph.laplacian(self.A_own, normed=True, return_diag=True)
        self.D = np.array(sum(self.A_own), dtype=float)
        self.A_sparse = dok_matrix(self.A_own)
        self.A_own = block_matrix_splitter(self.A_own, name=self.name, diag=False, block_size=block_size)
        print(1)

    def load_test_data_memmap(self, size):
        D = np.memmap("matrices/D.npy")
        self.A_own = self.generate_random_adjacency_matrix(size, if_sparse=False)
        self.vol = float(self.A_own.sum())
        # self.L, self.D = csgraph.laplacian(self.A_own, normed=True, return_diag=True)
        self.D = np.array(sum(self.A_own), dtype=float)

    def random_walk_sparse(self, pid, m_, T, num_workers, num_parties=2):
        np.random.seed(pid)
        matrix = sp.lil_matrix((self.total_node, self.total_node))

        for i in tqdm(range(self.vol * m_ // (num_workers * num_parties))):
            u, v = self.edges[i % self.vol]
            if not self.is_directed and np.random.rand() > 0.5:
                v, u = u, v
            for r in range(1, int(T + 1)):
                u_, v_, zp = self.path_sampling_sparse(u, v, r)
                matrix[u_, v_] += 2 * r / T / m_ / zp
        # self.A_sparse = matrix.tocsr()
        # self.L_sparse = sp.csgraph.laplacian(self.A_sparse, normed=False, return_diag=False)
        return matrix.tocsr()

    def calculate_L_sparse(self, A_res):
        self.A_sparse = A_res
        self.L_sparse = sp.csgraph.laplacian(self.A_sparse, normed=False, return_diag=False)

    def path_sampling_sparse_o(self, u, v, r):
        # sample a r-length path from edge(u, v) and return path end node
        def walk(adjacency_dict, start_node, walk_steps):
            """
            display walk operation
            :param adjacency_dict:
            :param start_node:
            :param walk_steps:
            :return: walk ending node and walk path
            """
            current_node = start_node
            walk_path = [start_node]
            for step in range(walk_steps):
                if start_node not in list(adjacency_dict.keys()):
                    # if current node have no out degree
                    return current_node, walk_path
                # display walk
                current_node = adjacency_dict[start_node][np.random.randint(len(adjacency_dict[start_node]))]
                walk_path.append(current_node)
                # update
                start_node = current_node
            return current_node, walk_path

        # if not A.issparse():
        #     print("Matrix A is not sparse matrix")
        # elif not A.isspmatrix_dok():
        #     A = A.todok()
        A = self.A_sparse.todok()


        # random select two integer number
        # r = random.sample(list(range(1, T + 1)), 1)[0]
        k = np.random.randint(r) + 1

        u0, u_path = walk(self.neighbors, u, k - 1)
        ur, v_path = walk(self.neighbors, v, r - k)

        # u_path.reverse()
        # p = u_path + v_path
        p = [(u_path[i], u_path[i + 1]) for i in range(len(u_path) - 1)] + [(u, v)] + [(v_path[i], v_path[i + 1])
                                                                                       for i in
                                                                                       range(len(v_path) - 1)]
        # print(p, u_path, v_path, u, v)
        r = len(p)

        Zp = 0
        ss = 0
        for path_node_index in range(len(p)):
            # path_node_pairs = (p[path_node_index], p[path_node_index+1])
            path_node_pairs = p[path_node_index]
            path_weight = A[path_node_pairs]
            ss += 1
            # calculate Zp
            Zp += 2 / path_weight

        # sparse_weight = (2 * r * self.total_node) / m_ * Zp
        # A_sparse[u0, ur] = sparse_weight
        return u0, ur, Zp


        # k = np.random.randint(r) + 1
        # zp, rand_u, rand_v = 2.0 / 1, k - 1, r - k
        # for i in range(rand_u):
        #     new_u = self.neighbors[u][alias_draw(self.alias_nodes[u][0], self.alias_nodes[u][1])]
        #     zp += 2.0 / 1
        #     u = new_u
        # for j in range(rand_v):
        #     new_v = self.neighbors[v][alias_draw(self.alias_nodes[v][0], self.alias_nodes[v][1])]
        #     zp += 2.0 / 1
        #     v = new_v
        # return u, v, zp

    def path_sampling_sparse(self, u, v, r):
        k = np.random.randint(r) + 1
        zp, rand_u, rand_v = 2.0 / 1, k - 1, r - k
        for i in range(rand_u):
            # new_u = u
            # zp += 0
            if self.neighbors[u]:
            # print(neighbors[u])
                new_u = self.neighbors[u][alias_draw(self.alias_nodes[u][0], self.alias_nodes[u][1])]
                zp += 2.0 / 1
                u = new_u
        for j in range(rand_v):
            # new_v = v
            # zp += 0
            if self.neighbors[v]:
                new_v = self.neighbors[v][alias_draw(self.alias_nodes[v][0], self.alias_nodes[v][1])]
                zp += 2.0 / 1
                v = new_v
        return u, v, zp


    def path_sampling(self, A, T,m, m_, if_block, block_size):
        """
        path sampling method for fednetsmf
        and compute L and D of the sparse graph
        :param T: window size
        :param m: non-zero number
        :return:
        """
        self.A_sparse = path_sampling(A, T, m, m_)
        self.A_sparse = self.A_sparse.todense()
        self.D_sparse = np.array(sum(self.A_sparse), dtype=float)
        self.L_sparse = np.diag(self.D_sparse) - self.A_sparse

        if if_block:
            # transform the big matrices into blocks
            self.A_sparse = block_matrix_splitter(self.A_sparse, name=f"A_sparse_{self.name}", diag=False, block_size=block_size)
            self.D_sparse = block_matrix_splitter(self.D_sparse, name=f"D_sparse_{self.name}", diag=True, block_size=block_size)
            self.L_sparse = block_matrix_splitter(self.L_sparse, name=f"L_sparse_{self.name}", diag=False, block_size=block_size)

    def path_sampling_test(self, A, T, m, if_block, block_size):
        """
        path sampling method for fednetsmf test
        and compute L and D of the sparse graph
        :param T: window size
        :param m: non-zero number
        :return:
        """
        # self.A_sparse = path_sampling(A, T, m)
        self.A_sparse = self.A_sparse.todense()
        # print(self.A_sparse)

        self.D_sparse = np.array(sum(self.A_sparse), dtype=float)
        self.L_sparse = np.diag(self.D_sparse) - self.A_sparse

        if if_block:
            # transform the big matrices into blocks
            self.A_sparse = block_matrix_splitter(self.A_sparse, name=f"A_sparse_{self.name}", diag=False, block_size=block_size)
            self.D_sparse = block_matrix_splitter(self.D_sparse, name=f"D_sparse_{self.name}", diag=True, block_size=block_size)
            self.L_sparse = block_matrix_splitter(self.L_sparse, name=f"L_sparse_{self.name}", diag=False, block_size=block_size)

    def calculate_M(self, D, O):
        """
        calculate and mask M matrix
        :param D: Degree diag matrix of complete graph
        :param O: Mask matrix generated by mask server
        :return:
        """
        self.M_sparse = np.dot(np.diag(D ** -1), np.add(self.D, -self.L_sparse)).dot(np.diag(D ** -1)) # calculate M
        self.M_sparse[self.M_sparse <= 1] = 1
        self.M_sparse = np.log(self.M_sparse)
        self.M_sparse = np.dot(self.M_sparse, O) # calcluate M*O get mask of M

    def calculate_M_block(self, D_, O, block_size):
        """
        calculate and mask M matrix (input and output block matrix)
        :param D_: (block) Degree diag matrix ** -1 of complete graph
        :param O: (block) Mask matrix generated by mask server
        :return:
        """
        tmp1 = diag_block_add(self.L_sparse, self.D_sparse, "tmp1", "-+")
        tmp2 = diag_block_dot(tmp1, D_, name="tmp2", dot_side="left")
        tmp3 = diag_block_dot(tmp2, D_, name="tmp3", dot_side="right")
        self.M_sparse = block_dot(tmp3, O, block_size=block_size, name=f"M_sparse_{self.name}")

    def calculate_M_sparse(self, D_inv, O, vol, b):
        """
        calculate and mask M matrix (input and output block matrix)
        :param D_: (block) Degree diag matrix ** -1 of complete graph
        :param O: (block) Mask matrix generated by mask server
        :return:
        """
        # print(self.D, self.L_sparse)
        self.M_sparse = D_inv.dot(self.D - self.L_sparse).dot(D_inv).dot(O)
        # self.M_sparse = tmp1.dot(O)
        self.M_sparse = self.M_sparse * vol / b


        # self.M_sparse.data[self.M_sparse.data <= 1] = 1
    def calculate_M_sparse_test(self, D_inv, O, vol, b):
        """
        calculate and mask M matrix (input and output block matrix)
        :param D_: (block) Degree diag matrix ** -1 of complete graph
        :param O: (block) Mask matrix generated by mask server
        :return:
        """
        # print(self.D, self.L_sparse)
        self.M_sparse = D_inv.dot(self.D - self.L_sparse).dot(D_inv)
        # self.M_sparse = tmp1.dot(O)
        self.M_sparse = self.M_sparse * vol / b

        self.M_sparse[self.M_sparse <= 1] = 1
        self.M_sparse = np.log(self.M_sparse.toarray())

    def calculate_M_sparse_base(self, D_inv, vol=0, b=1):
        self.M_sparse = D_inv.dot(self.D - self.L_sparse).dot(D_inv)

        self.M_sparse = self.M_sparse * vol / b
        self.M_sparse.data[self.M_sparse.data <= 1] = 1
        self.M_sparse.data = np.log(self.M_sparse.data)

    def calculate_M_sparse_iterate(self,M, O):
            self.M_iterate = safe_sparse_dot(M, O)


    def compute_B_block(self, Q, block_size):
        self.B = block_dot(list(map(list, zip(*Q))), self.M_sparse, name=f"B_{self.name}", block_size=block_size)


    def compute_B_sparse(self, Q):
        self.B = safe_sparse_dot(Q.T, self.M_sparse)









    def get_DA(self, D):
        self.DA_own = D.power(-1).dot(self.A_own)
        # self.DA_own = np.dot(np.diag(D.power(-1)), self.A_own)

    def get_DA_block(self, D):
        self.DA_own = diag_block_dot(self.A_own, D, "DA_" + self.name, dot_side="left", inverse=True)

    def load_data(self, path):
        '''
        load .mat sparse matrices
        :param path:
        :return:
        '''
        data = scipy.io.loadmat(path)
        logger.info("loading mat file %s", path)
        # print(data.keys())
        # print(data['Problem'][0][0][2])
        # matrix = data['Problem'][0][0][2]
        matrix = data['network']
        self.A_own = matrix
        # print(sum(self.A_own))
        self.vol = float(self.A_own.sum())
        self.D = np.array(sum(self.A_own), dtype=float)
    def load_data_block(self, path, block_size):
        data = scipy.io.loadmat(path)
        logger.info("loading mat file %s", path)
        # print(data.keys())
        print(data['network'].shape)
        matrix = data['network'].todense()

        # print(data['Problem'][0][0][2].shape)
        # matrix = data['Problem'][0][0][2].todense()
        self.A_own = matrix
        # print(sum(self.A_own))
        self.vol = float(self.A_own.sum())
        self.D = np.array(sum(self.A_own), dtype=float)
        self.A_own = block_matrix_splitter(self.A_own, name=self.name, diag=False, block_size=block_size)

    def load_data_edgelist(self, path, blocksize):
        info_list = path[:-4].split("_")[1:]
        print(info_list)
        num_part = int(info_list[0])
        total_part = int(info_list[1])
        total_node = int(info_list[2])
        total_edge = int(info_list[3])
        self.A_sparse = dok_matrix((total_node, total_node))
        with open(path, "r") as fr:
            edgelist = fr.readlines()
        for edges in edgelist:
            m, n = edges.split(" ")
            m, n = int(m), int(n)
            self.A_sparse[m,n] = 1
        self.vol = len(edgelist)
        self.D = np.array(sum(self.A_sparse.todense()), dtype=float)
        self.A_own = block_matrix_splitter(self.A_sparse, name=self.name, diag=False, block_size=blocksize)

    def load_data_edgelist_sparse(self, path, is_directed=False):
        info_list = path[:-4].split("_")[1:]
        print(info_list)
        num_part = int(info_list[0])
        total_part = int(info_list[1])
        self.total_node = int(info_list[2])
        total_edge = int(info_list[3])
        self.A_sparse = dok_matrix((self.total_node, self.total_node))

        with open(path, "r") as fr:
            self.edgelist = [[int(edges.split(" ")[0]),int(edges.split(" ")[1])] for edges in fr.readlines()]
        for edges in self.edgelist:
            # m, n = edges.split(" ")
            m, n = edges
            self.A_sparse[m,n] = 1

        # self.neighbors1 = []
        # A_paths = list(dict(self.A_sparse).keys())
        # for path in A_paths:
        #     if path[0] not in list(self.neighbors1):
        #         self.neighbors1[path[0]] = [path[1]]
        #     self.neighbors1[path[0]].append(path[1])

        self.A_sparse = self.A_sparse.tocsr()

        if is_directed:
            self.G = nx.DiGraph(self.A_sparse)
        else:
            self.G = nx.Graph(self.A_sparse)

        node2id = dict([(node, vid) for vid, node in enumerate(self.G.nodes())])
        self.is_directed = nx.is_directed(self.G)

        self.num_node = self.G.number_of_nodes()
        self.num_edge = self.G.number_of_edges()

        print("!!!!",len(self.edgelist), self.num_edge)
        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges()]

        id2node = dict(zip(node2id.values(), node2id.keys()))

        self.num_neigh = np.asarray([len(list(self.G.neighbors(id2node[i]))) for i in range(self.num_node)])
        self.neighbors = [[node2id[v] for v in self.G.neighbors(id2node[i])] for i in range(self.num_node)]
        self.alias_nodes = {}
        self.node_weight = {}
        for i in range(self.num_node):
            unnormalized_probs = [self.G[id2node[i]][nbr].get("weight", 1.0) for nbr in self.G.neighbors(id2node[i])]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[i] = alias_setup(normalized_probs)
            self.node_weight[i] = dict(
                zip([node2id[nbr] for nbr in self.G.neighbors(id2node[i])], unnormalized_probs, ))



        self.vol = self.num_edge
        self.D = sp.diags(np.array(self.A_sparse.sum(axis=0))[0], format="csr")
        self.D_inv = self.D.power(-1)

    def calculate_size(self):
        return self.A_own.shape[0]

    def get_residual_1(self, residual):
        self.residual_1 = residual

    def get_residual_2(self, residual):
        self.residual_2 = residual

    def calculate_residual_1(self):
        self.residual_1.append(self.DA_own - self.random_matrices[0])
        self.residual_1.append(self.DA_own - self.random_matrices[1])
        return self.residual_1

    def calculate_residual_1_block(self):
        self.residual_1.append(
            diag_block_add(self.DA_own, self.random_matrices[0], "residual_1_0_" + self.name, negative="+-"))
        self.residual_1.append(
            diag_block_add(self.DA_own, self.random_matrices[1], "residual_1_1_" + self.name, negative="+-"))
        return self.residual_1

    def calculate_residual_2(self):
        self.residual_2.append(self.DA_own - self.random_matrices[0])
        self.residual_2.append(self.DA_own - self.random_matrices[1])
        return self.residual_2

    def calculate_residual_2_block(self):
        self.residual_2.append(
            diag_block_add(self.DA_own, self.random_matrices[0], "residual_2_0_" + self.name, negative="+-"))
        self.residual_2.append(
            diag_block_add(self.DA_own, self.random_matrices[1], "residual_2_1_" + self.name, negative="+-"))
        return self.residual_2

    def calculate_PDAQ(self, P, D, Q):
        # self.PDAQ = np.dot(P,sparse.diags(D**-1)).dot(self.A_own).dot(Q)
        self.PDAQ = P.dot(D.power(-1)).dot(self.A_own).dot(Q)

    def calculate_PDAQ_block(self, P, D, Q):
        tmp1 = diag_diag_block_dot(P, D, "tmp1", dot_side="right", inverse=True)
        tmp2 = diag_block_dot(self.A_own, tmp1, "tmp2", dot_side="left")
        self.PDAQ = diag_block_dot(tmp2, Q, "PDAQ_" + self.name, dot_side="right")
        return self.PDAQ

    def calculate_PDA2Q(self, P, D, Q):
        # DA = np.dot(sparse.diags(D**-1),self.A_own)
        DA = D.power(-1).dot(self.A_own)
        # DA = np.dot(np.diag(D ** -1), self.A_own)
        DA2 = np.dot(DA, DA)
        self.PDA2Q = np.dot(P, DA2).dot(Q)

    # Alice
    def get_random_matrices(self, random_matrices):
        self.random_matrices = random_matrices

    def get_T(self, T):
        self.T = T

    def calculate_W(self):
        self.T_ = random_matrix(self.calculate_size())
        return np.dot(self.random_matrices[0], self.residual_2[1]) \
               + np.dot(self.residual_2[0], self.random_matrices[1]) \
               + np.dot(self.DA_own, self.DA_own) \
               - self.T_

    def calculate_W_block(self, size, block_size):
        self.T_ = generate_block_random_martic(size, block_size, "T_")
        tmp1 = diag_block_dot(self.residual_2[1], self.random_matrices[0], "tmp1", dot_side="left")
        tmp2 = diag_block_dot(self.residual_2[0], self.random_matrices[1], "tmp2", dot_side="right")
        # show_block_matrix(tmp2, block_size, shape=(size,size))
        tmp3 = block_dot(self.DA_own, self.DA_own, block_size, "tmp3",(size,size))
        tmp4 = block_add(tmp1, tmp2, block_size, "tmp4")
        tmp5 = block_add(tmp3, tmp4, block_size, "tmp5")
        return diag_block_add(tmp5, self.T_, "W", negative="+-")

    def output_Ts(self):
        # return sparse.csc_matrix(self.T+self.T_)
        return self.T + self.T_
    def output_Ts_block(self):
        return diag_diag_block_add(self.T, self.T_, "Ts", "++")

    def output_Ts_block(self):
        return diag_diag_block_add(self.T, self.T_, "Ts", negative="++")

    # Bob
    def get_C(self, C):
        # self.C = sparse.csc_matrix(C)
        self.C = C

    def get_W(self, W):
        # self.W = sparse.csc_matrix(W)
        self.W = W

    def output_U(self):
        a = np.dot(self.residual_1[0], self.DA_own)
        b = np.dot(self.DA_own, self.residual_1[1])
        c = np.dot(self.DA_own, self.DA_own)
        d = np.add(np.add(a, b), c)
        res = np.add(d, self.W)
        res = np.add(res, self.C)
        return res

    def output_U_block(self, block_size, size):
        tmp1 = block_dot(self.residual_1[0], self.DA_own, block_size, "tmp1",(size,size))
        tmp2 = block_dot(self.DA_own, self.residual_1[1], block_size, "tmp2",(size,size))
        tmp3 = block_dot(self.DA_own, self.DA_own, block_size, "tmp3",(size,size))
        tmp4 = block_add(tmp1, tmp2, block_size, "tmp4")
        tmp5 = block_add(tmp3, tmp4, block_size, "tmp5")
        tmp6 = block_add(tmp5, self.W, block_size, "tmp6")
        return diag_block_add(tmp6, self.C, "U", "++")

    def calculate_A2(self,P, D, DQ, block_size, size):
        A2 = block_dot(self.A_own, self.A_own, block_size, "A2", (size,size))
        tmp1 = diag_diag_block_dot(P, D, "tmp1", dot_side="right", inverse=True)
        tmp2 = diag_block_dot(A2, tmp1, "tmp2", dot_side="left")
        return diag_block_dot(tmp2, DQ, "A2_" + self.name, dot_side="right")