import numpy as np
from utils import *

class FactorizationServer:
    def __init__(self):
        super(FactorizationServer, self).__init__()
        self.masked_matrix = []

        self.M = []


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

    def calculate_M_block_multi_party(self, PDAQs, PDAAQs, As, vol, T, b, block_size):
        for i in range(len(PDAQs)):
            if i==0:
                tmp1 = PDAQs[0]
            else:
                tmp1 = block_add(tmp1, PDAQs[i], block_size, "tmp1")

        for i in range(len(PDAAQs)):
            tmp1 = block_add(tmp1, PDAAQs[i], block_size, "tmp1")

        for i in range(len(As)):
            tmp1 = block_add(tmp1, As[i], block_size, "tmp1", negative="++")
        S = vol / (T * b)
        self.M = block_constant_multiply(S, tmp1, "M")
        return self.M



    def calculate_M_line(self,PDAQ1,PDAQ2,vol,b):
        S = vol/b
        self.M = S*(PDAQ1+PDAQ2)
        return self.M
    def calculate_M_line_block(self, PDAQ1, PDAQ2, vol, b, block_size):
        S = vol/b
        tmp1 = block_add(PDAQ1,PDAQ2,block_size,"tmp1")
        self.M = block_constant_multiply(S, tmp1, "M")
