import numpy as np
from utils import *

M = []
block_size = 2108
for i in range(36):
    M.append([])
    for j in range(36):
        M[i].append("M_"+str(i)+"_"+str(j)+"_"+str(block_size)+".npz")
Mreal = show_block_matrix(M,block_size)
U,S,vT = svd(M)
