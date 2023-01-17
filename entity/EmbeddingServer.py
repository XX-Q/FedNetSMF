import gc

import numpy as np
from utils import *

class EmbeddingServer:
    def __init__(self):
        super(EmbeddingServer, self).__init__()
        self.masked_matrix = []

        self.M = []

        self.M_sum = []
        self.Q = []

        self.B_sum = []


    def receive_masked_matrix(self, masked_matrix):
        self.masked_matrix = masked_matrix

    def calculate_M(self, PDAQ1, PDAQ2, PDAAQs, vol, T, b):
        S = vol/(T*b)
        self.M = S*(PDAQ1 + PDAQ2 +  PDAAQs)
        return self.M
    def calculate_M_block(self, PDAQ1, PDAQ2, PDAAQs, vol, T, b, block_size):
        S = vol/(T*b)
        tmp1 = block_add(PDAQ1, PDAQ2, block_size, "tmp1")
        tmp2 = block_add(tmp1, PDAAQs, block_size, "tmp2")
        self.M = block_constant_multiply(S, tmp2, "M")
        return self.M
    def calculate_M_line(self,PDAQ1,PDAQ2,vol,b):
        S = vol/b
        self.M = S*(PDAQ1+PDAQ2)
        return self.M
    def calculate_M_line_block(self, PDAQ1, PDAQ2, vol, b, block_size):
        S = vol/b
        tmp1 = block_add(PDAQ1,PDAQ2,block_size,"tmp1")
        self.M = block_constant_multiply(S, tmp1, "M")


    def calculate_M_sum(self, Ms, block_size):
        self.M_sum = Ms[0]
        for M_index in range(1, len(Ms)):
            self.M_sum = block_add(self.M_sum, Ms[M_index], block_size=block_size, name="M_sum")
            # TODO change the method into secureAgg

    def calculate_qr_result(self, block_size):
        self.Q = qr_block_matrix(self.M_sum, block_size=block_size, name="Q")
        M_ = show_block_matrix(self.M_sum, block_size)
        Q_ = show_block_matrix(self.Q, block_size)
        print("_")

    def calculate_B_sum(self, Bs, block_size):
        self.B_sum = Bs[0]
        for M_index in range(1, len(Bs)):
            self.B_sum = block_add(self.B_sum, Bs[M_index], block_size=block_size, name="B_sum")

            # TODO change the method into secureAgg

    def calculate_svd_result(self, block_size):
        self.B_sum = show_block_matrix(self.B_sum, block_size)
        U, s, _ = svd(self.B_sum)
        del _
        gc.collect()

        s = block_matrix_splitter(s, "s", block_size=block_size, diag=True)
        U = block_matrix_splitter(U, "U", block_size=block_size, diag=False)
        U = block_dot(self.Q, U, block_size, name="U")
        embedding_result = diag_block_dot(U, s, "embedding_result", dot_side="right", block_size=block_size)
        embedding_result = show_block_matrix(embedding_result, block_size=block_size)
        return embedding_result






