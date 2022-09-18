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
    def calculate_M_line(self,PDAQ1,PDAQ2,vol,b):
        S = vol/b
        self.M = S*(PDAQ1+PDAQ2)
        return self.M
    def calculate_M_line_block(self, PDAQ1, PDAQ2, vol, b, block_size):
        S = vol/b
        tmp1 = block_add(PDAQ1,PDAQ2,block_size,"tmp1")
        self.M = block_constant_multiply(S, tmp1, "M")
