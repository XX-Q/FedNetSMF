import numpy as np

from utils import *

def power_iteration(A, Omega, power_iter = 3):
    Y = A @ Omega
    for q in range(power_iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    return Q

def rsvd(A, Omega):
    Q = power_iteration(A, Omega)
    B = Q.T @ A
    u_tilde, s, v = np.linalg.svd(B, full_matrices = 0)
    u = Q @ u_tilde
    return u, s, v


import numpy as np
import imageio

image = imageio.imread('test.png')
A = image[:, :, 1]
u, s, v = np.linalg.svd(A, full_matrices = 0)



import matplotlib.pyplot as plt

rank = 20
Omega = np.random.randn(A.shape[1], rank)

Omega = random_matrix_unblock(A.shape[1], rank)

u, s, v = rsvd(A, Omega)

print(u)

plt.subplot(1, 2, 2)
plt.imshow(u[:, : rank] @ np.diag(s[: rank]) @ v[: rank, :])
plt.title('The rank-50 Lena using randomized SVD.')
plt.axis('off')
plt.show()