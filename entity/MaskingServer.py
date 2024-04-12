import numpy as np
from scipy import sparse

from utils import *

class MaskingServer():
    def __init__(self):
        super(MaskingServer, self).__init__()
        self.size = 0

        self.T = []
        self.P = []
        self.Q = []
        self.DQ = []
        self.C = []
        self.O = []

        self.random_matrices_1 = []
        self.random_matrices_2 = []


    def get_size(self, size):
        self.size = size

    def generate_random_matrix_T(self):
        self.T = random_matrix(self.size)
    def generate_ramdom_matrix_T_block(self, block_size):
        self.T = generate_block_random_martic(self.size, block_size, "T")

    def generate_random_matrices(self):
        for i in range(2):
            self.random_matrices_1.append(random_matrix(self.size))
            self.random_matrices_2.append(random_matrix(self.size))
    def generate_random_matrices_block(self, block_size):
        for i in range(2):
            self.random_matrices_1.append(generate_block_random_martic(self.size, block_size, "random_matrices_1_"+str(i)))
            self.random_matrices_2.append(
                generate_block_random_martic(self.size, block_size, "random_matrices_2_" + str(i)))

    def generate_random_matrix_O(self):
        self.O = random_matrix(self.size)
    def generate_ramdom_matrix_O_block(self, block_size):
        self.O = generate_block_random_martic(int(self.size / block_size), block_size, "O")


    def calculate_C(self):
        self.C = np.dot(self.random_matrices_1[0],self.random_matrices_2[1])\
        +np.dot(self.random_matrices_2[0],self.random_matrices_1[1]) - self.T
    def calculate_C_block(self):
        tmp1 = diag_diag_block_dot(self.random_matrices_1[0], self.random_matrices_2[1],"tmp1",dot_side="right")
        tmp2 = diag_diag_block_dot(self.random_matrices_2[0], self.random_matrices_1[1],"tmp2", dot_side="right")
        tmp3 = diag_diag_block_add(tmp1, tmp2, "tmp3", negative="++")
        self.C = diag_diag_block_add(tmp3, self.T, "C", negative="+-")


    def generate_orthogonal_matrices(self, D):
        self.P = orthogonal_matrix(self.size)
        self.Q = orthogonal_matrix(self.size)
        # self.DQ = np.dot(sparse.diags(D ** -1),self.Q)
        self.DQ = D.power(-1).dot(self.Q)
        # self.DQ = np.dot(np.diag(D.power(-1)), self.Q)
        return self.Q
    def generate_orthogonal_matrices_block(self, D, block_size):
        p = self.P = orthogonal_matrix_block(self.size, block_size, "P")
        q = self.Q = orthogonal_matrix_block(self.size, block_size, "Q")
        self.DQ = diag_diag_block_dot(self.Q, D,"DQ", dot_side="left", inverse=True)
        return p, q


    def generate_random_matrix(self, n, d, block_size):
        self.O = random_matrix_unblock(n, d)
        self.O = block_matrix_splitter(self.O, "O", diag=False, block_size=block_size)

    def generate_random_matrix_sparse(self, n, d):
        self.O = random_matrix_unblock(n, d)

    def generate_random_matrix_test(self, n, d, block_size):
        self.O = [['matrices/O_0_0_100.npz'], ['matrices/O_1_0_100.npz'], ['matrices/O_2_0_100.npz'], ['matrices/O_3_0_100.npz'], ['matrices/O_4_0_100.npz'], ['matrices/O_5_0_100.npz'], ['matrices/O_6_0_100.npz'], ['matrices/O_7_0_100.npz'], ['matrices/O_8_0_100.npz'], ['matrices/O_9_0_100.npz']]



