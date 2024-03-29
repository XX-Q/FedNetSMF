import numpy as np
import os
import shutil
import gc

import scipy.sparse
from scipy.sparse import csgraph
from scipy.sparse import dok_matrix
import random


data_dir = "dataset"


def svd(x):
    m, n = x.shape
    if m >= n:
        return np.linalg.svd(x)
    else:
        u, s, v = np.linalg.svd(x)
        return v.T, s, u.T

def power_iteration(A, Omega, power_iter = 3):
    Y = np.dot(A, Omega)
    for q in range(power_iter):
        Y = np.dot(A, np.dot(A.T, Y))
    Q, _ = np.linalg.qr(Y)
    return Q

def rsvd(A, O):
    """
    rnadomized svd
    :param A: matrix to be svd (n x n)
    :param O: random matrix (n x m)
    :return:
    """
    Q = power_iteration(A, O)
    # Y = np.dot(A, O)
    # Q, _ = np.linalg.qr(Y)
    B = np.dot(Q.T, A)
    u_tilde, s, v = np.linalg.svd(B, full_matrices=False)
    u = np.dot(Q, u_tilde)
    return u, s, v


def random_matrix(size):
    """
    生成指定大小的随机矩阵，元素为整数
    :param size: size of random matrix
    :return: random matrix
    """
    return np.random.randint(low=0, high=10, size=(size, size)) + np.random.random(size=(size,size))

def random_matrix_unblock(row, col):
    """
    生成指定大小的随机矩阵，元素为整数
    :param size: size of random matrix
    :return: random matrix
    """
    return np.random.randint(low=0, high=10, size=(row, col)) + np.random.random(size=(row, col))


def load_mnist(n=5000):
    data = np.load(os.path.join(data_dir, 'mnist', 'mnist.npz'))
    x_train = data['x_train'].reshape([-1, 28 * 28])
    y_train = data['y_train']
    x_test = data['x_test'].reshape([-1, 28 * 28])
    y_test = data['y_test']
    x = np.concatenate([x_train, x_test], axis=0).astype(float)
    y = np.concatenate([y_train, y_test], axis=0).astype(float)
    return x[:n].T, y[:n]


def orthogonal_matrix(size):
    q, r = np.linalg.qr(np.random.randn(size, size), mode='full')
    return q

def qr_block_matrix(M, name, block_size):
    """
    run qr decomposition on block matrix and return result in block
    :param M: block matrix
    :return: Q matrix in block
    """
    M = show_block_matrix(M, block_size=block_size)
    q, r = np.linalg.qr(M, mode='reduced')
    q = block_matrix_splitter(q, name=name, block_size=block_size, diag=False)
    return q






# def random_matrix(size):
#     return np.random.random([size,size])

def secure_aggregation(xs):
    """
    secure aggregation的方法，输入是不同dataholder拥有的x序列
    1.
    :param xs: [x1, x2, ..., xi] 是x的小矩阵
    :return: secure aggregation 的x'
    """
    k = len(xs)
    x_size = xs[0].shape  # 假设所有x的大小一致

    pertubations = []

    for i in range(k):
        pairs = []
        for j in range(k):
            pairs.append(random_matrix(x_size))  # 对于每一个xi，生成其对于其余xj的扰动
        pertubations.append(pairs)  # :shape [ki, kj, m, n]

    pertubations = np.array(pertubations)
    pertubations -= np.transpose(pertubations, axes=[1, 0, 2, 3])  # ri- rj

    ys = []
    for i in range(k):
        ys.append(xs[i] + np.sum(pertubations[i], axis=0))

    return np.sum(ys, axis=0)


def matrix_splitter(matrix, num_dataholders, mode=0):
    """
    输入矩阵（一般是Q或X），以及dataholder的个数，按照dataholder的数目返回分割后的矩阵
    :param matrix: Q或X矩阵
    :param num_dataholders: Dataholder的个数
    :return: Qs或Xs小矩阵序列
    """
    m, n = matrix.shape
    Ms = []
    if mode == 0:
        col = n // num_dataholders
        for i in range(0, num_dataholders):
            Ms.append(matrix[:, i * col:(i + 1) * col])
    elif mode == 1:
        col = m // num_dataholders
        for i in range(0, num_dataholders):
            Ms.append(matrix[i * col:(i + 1) * col, :])

    return Ms


def test_netmf(A,D , T, b, P, Q):
    '''
    test the result of netmf
    :param A:
    :param T:
    :param b:
    :param P:
    :param Q:
    :return:
    '''
    vol = float(A.sum())
    S = vol / (T * b)
    # L, D = csgraph.laplacian(A, normed=True, return_diag=True)
    D_rt = np.diag(D ** -1)
    P1 = np.dot(D_rt, A)
    P2 = np.dot(P1, P1)
    return S * np.dot(P, P1 + P2).dot(np.dot(D_rt, Q))

def test_line(A, D, b, P, Q):
    vol = float(A.sum())
    S = vol / b
    # L, D = csgraph.laplacian(A, normed=True, return_diag=True)
    D_rt = np.diag(D ** -1)
    P1 = np.dot(D_rt, A)
    return S * np.dot(P, P1).dot(np.dot(D_rt, Q))

def find_block_size(size):
    for i in range(100, size):
        if size % i == 0:
            return i


def block_matrix_splitter(matrix, name, block_size, diag=False):
    if not diag:
        size_row, size_col = matrix.shape[0], matrix.shape[1]

        res_size_row, res_size_col = int(size_row / block_size), int(size_col / block_size)
        res = []
        for i in range(0, res_size_row):
            res.append([])
            for j in range(0, res_size_col):
                matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(block_size) + ".npz"
                # matrix = scipy.sparse.csc_matrix(matrix[i*block_size:(i+1)* block_size, j*block_size:(j+1)* block_size])
                scipy.sparse.save_npz(matrix_name,scipy.sparse.csc_matrix(matrix[i*block_size:(i+1)* block_size, j*block_size:(j+1)* block_size]))
                # np.save(matrix_name, matrix[i*block_size:(i+1)* block_size, j*block_size:(j+1)* block_size])
                res[i].append(matrix_name)
    else:
        matrix = np.array(matrix)
        matrix = matrix.reshape((-1))
        size = len(matrix)
        res_size = int(size / block_size)

        res = []
        for i in range(0, res_size):
            matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(i) + '_' + str(block_size) + ".npz"
            scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(np.diag(matrix[i*block_size:(i+1)* block_size])))
            # np.save(matrix_name, np.diag(matrix[i*block_size:(i+1)* block_size]))
            res.append(matrix_name)
    return res


# def block_dot(m1, m2, block_size, name):
#     res_size = len(m1)
#     # print(res_size)
#     res = [[""] * res_size] * res_size
#     for i in range(res_size):
#         for j in range(res_size):
#             ijres = np.zeros(shape=(block_size, block_size))
#             for k in range(res_size):
#                 b1 = np.load(m1[i][k])
#                 b2 = np.load(m2[k][j])
#                 bres = np.dot(b1, b2)
#                 ijres += bres
#             matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
#                 block_size) + ".npz"
#             np.save(matrix_name, ijres)
#             res[i][j] = matrix_name
#     return res

def block_dot(A, B, block_size, name):
    a = len(A)
    b = len(B)
    c = len(B[0])

    # Final = [[""] * a] * a
    Final = []
    for i in range(0, a):
        Final.append([])
        for j in range(0, c):
            sum = np.zeros(shape=(block_size, block_size))
            for h in range(0, b):
                # b1 = np.load(A[i][h])
                b1 = scipy.sparse.load_npz(A[i][h]).todense()
                # b2 = np.load(B[h][j])
                b2 = scipy.sparse.load_npz(B[h][j]).todense()
                sum = np.add(sum,np.dot(b1,b2))
            matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
                block_size) + ".npz"
            scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(sum))
            # np.save(matrix_name, sum)
            Final[i].append(matrix_name)
    return Final

    # Final = []
    # for i in range(0, a):
    #     Final.append([])
    #     for j in range(0, a):
    #         sum = np.zeros(shape=(block_size, block_size))
    #         for h in range(0, a):
    #             # b1 = np.load(A[i][h])
    #             b1 = scipy.sparse.load_npz(A[i][h]).todense()
    #             # b2 = np.load(B[h][j])
    #             b2 = scipy.sparse.load_npz(B[h][j]).todense()
    #             sum = np.add(sum, np.dot(b1, b2))
    #         matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
    #             block_size) + ".npz"
    #         scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(sum))
    #         # np.save(matrix_name, sum)
    #         Final[i].append(matrix_name)
    # return Final

def orthogonal_matrix_block(size, block_size, name):
    res_size = int(size/block_size)
    res = []
    for i in range(res_size):
        q, r = np.linalg.qr(np.random.randn(block_size, block_size), mode='full')
        matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(i) + '_' + str(
            block_size) + ".npz"
        scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(q))
        # np.save(matrix_name,q)
        res.append(matrix_name)
    return res


def block_mask(m, block_size, mask_name, mask_side="left"):
    '''
    use block matrix to mask target matrix
    :param m: target matrix
    :param block_size: block size
    :return:
    '''
    num_block = len(m)
    mask_matrix_block = []
    for i in range(num_block):
        mask_matrix = orthogonal_matrix(block_size)
        matrix_name = "matrices/" + mask_name + "_" + str(i) + '_' + str(i) + '_' + str(
            block_size) + ".npz"
        mask_matrix_block.append(matrix_name)
        scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(mask_matrix))
        # np.save(matrix_name, mask_matrix)

        for j in range(len(m[i])):
            # matrix_to_mask = np.load(m[i][j])
            matrix_to_mask = scipy.sparse.load_npz(m[i][j]).todense()
            if mask_side=="left":
                masked_matrix = np.dot(mask_matrix, matrix_to_mask)
            else:
                masked_matrix = np.dot(matrix_to_mask,mask_matrix)
            scipy.sparse.save_npz(m[i][j], scipy.sparse.csc_matrix(mask_matrix))
            # np.save(m[i][j], masked_matrix)
    return mask_matrix_block

def block_demask(m, mask_matrix,name,block_size, demask_side="left"):
    '''
    demask block matrices on block diag
    :param m:
    :param mask_matrix:
    :param demask_side:
    :return:
    '''
    num_block = len(m)
    res = []
    for i in range(num_block):
        # demask_matrix = np.load(mask_matrix[i]).T
        demask_matrix = scipy.sparse.load_npz(mask_matrix[i]).todense().T
        res.append([])
        for j in range(num_block):
            # matrix_to_demask = np.load(m[i][j])
            matrix_to_demask = scipy.sparse.load_npz(m[i][j]).todense()
            if demask_side=="left":
                demasked_matrix = np.dot(demask_matrix, matrix_to_demask)
            else:
                demasked_matrix = np.dot(matrix_to_demask,demask_matrix)
            matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(i) + '_' + str(
                block_size) + ".npz"
            res[i].append(matrix_name)
            scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(demask_matrix))
    return res

def generate_block_random_martic(size, block_size, name):
    '''
    generate diag block random matric
    :param size: size of blocked matrix (real_size/block_size)
    :param block_size: block
    :param name: matrix name
    :return:
    '''
    res = []
    for i in range(size):
        matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(i) + '_' + str(
            block_size) + ".npz"
        rm = random_matrix(block_size)
        scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(rm))
        # np.save(matrix_name, rm)

        res.append(matrix_name)
    return res


def diag_block_dot(m, diag_martix, name, dot_side="left", inverse=False, block_size=0):
    res = []
    for i in range(len(m)):
        res.append([])
        for j in range(len(m[0])):
            res[i].append("")
    tmplen = len(m)
    # if dot_side == "right":
    #     tmplen = len(m[0])
    for i in range(tmplen):
        # rm = np.load(diag_martix[i])

        for j in range(len(m[0])):
            rm = scipy.sparse.load_npz(diag_martix[j]).todense()
            if inverse:
                rm = np.diag(np.diag(rm) ** -1)
            # block_size = len(rm)
            if dot_side== "left":
                # matrix = np.load(m[i][j])
                matrix = scipy.sparse.load_npz(m[j][i]).todense()
                dot_res = np.dot(rm, matrix)
                matrix_name = "matrices/" + name + "_" + str(j) + '_' + str(i) + '_' + str(
                    block_size) + ".npz"
                # np.save(matrix_name, dot_res)
                scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(dot_res))
                res[j][i] = matrix_name
            else:
                # matrix = np.load(m[j][i])
                matrix = scipy.sparse.load_npz(m[i][j]).todense()
                dot_res = np.dot(matrix, rm)
                matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
                block_size) + ".npz"
                scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(dot_res))
                # np.save(matrix_name, dot_res)
                res[i][j]= matrix_name
    return res

def diag_diag_block_dot(m, diag_martix, name, dot_side="left", inverse=False):
    res = []
    for i in range(len(m)):
        # rm = np.load(diag_martix[i])
        rm = scipy.sparse.load_npz(diag_martix[i]).todense()

        if inverse:
            rm = np.diag(np.diag(rm)**-1)
        block_size = len(rm)
        # matrix = np.load(m[i])
        matrix = scipy.sparse.load_npz(m[i]).todense()
        if dot_side == "left":
            dot_res = np.dot(rm, matrix)
        else:
            dot_res = np.dot(matrix, rm)
        matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(i) + '_' + str(
            block_size) + ".npz"
        # np.save(matrix_name, dot_res)
        scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(dot_res))
        res.append(matrix_name)
    return res

def block_constant_multiply(C, m, name):
    res = []
    for i in range(len(m)):
        res.append([])
        for j in range(len(m)):
            # matrix = np.load(m[i][j])
            matrix = scipy.sparse.load_npz(m[i][j]).todense()
            block_size = len(matrix)
            dot_res = C*matrix
            matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
                block_size) + ".npz"
            # np.save(matrix_name, dot_res)
            scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(dot_res))
            res[i].append(matrix_name)
    return res

def block_add(A, B, block_size, name):
    a = len(A)
    b = len(A[0])
    Final = []
    for i in range(0, a):
        Final.append([])
        for j in range(0, b):
            # b1 = np.load(A[i][j])
            b1 = scipy.sparse.load_npz(A[i][j]).todense()
            # b2 = np.load(B[i][j])
            b2 = scipy.sparse.load_npz(B[i][j]).todense()
            sum = np.add(b1, b2)
            matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
                block_size) + ".npz"
            # np.save(matrix_name, sum)
            scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(sum))
            Final[i].append(matrix_name)
    return Final

def diag_block_add(m, diag_martix, name, negative="++"):
    # TODO have not test
    res = []
    for i in range(len(m)):
        # rm = np.load(diag_martix[i])
        rm = scipy.sparse.load_npz(diag_martix[i]).todense()
        block_size = len(rm)
        res.append([])
        for j in range(len(m)):
            if i==j:
                # matrix = np.load(m[i][j])
                matrix = scipy.sparse.load_npz(m[i][j]).todense()
                if negative[0] == '-':
                    matrix = -matrix
                if negative[1] == "-":
                    rm = -rm
                add_res = np.add(matrix, rm)
                matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
                block_size) + ".npz"
                # np.save(matrix_name, add_res)
                scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(add_res))
            else:
                matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
                    block_size) + ".npz"
                shutil.copy(m[i][j], matrix_name)
            res[i].append(matrix_name)
    return res

def diag_diag_block_add(dm1, dm2, name, negative="++"):
    res = []
    for i in range(len(dm1)):
        # rm = np.load(dm2[i])
        rm = scipy.sparse.load_npz(dm2[i]).todense()
        block_size = len(rm)
        # matrix = np.load(dm1[i])
        matrix = scipy.sparse.load_npz(dm1[i]).todense()

        if negative[0]=="-":
            matrix= -matrix
        if negative[1]=="-":
            rm = -rm
        add_res = np.add(matrix, rm)
        matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(i) + '_' + str(
            block_size) + ".npz"
        # np.save(matrix_name, add_res)
        scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(add_res))
        res.append(matrix_name)
    return res

def show_block_matrix(matrix, block_size):
    # TODO 这里有时间改一下，默认矩阵都是方阵了
    if len(np.array(matrix).shape)==2:
        m = len(matrix)
        n = len(matrix[0])
        res = np.zeros(shape=(max(m,n)*block_size,max(m,n)*block_size))
        for i in range(m):
            for j in range(n):
                # res[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=np.load(matrix[i][j])
                res[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]=scipy.sparse.load_npz(matrix[i][j]).todense()
                # res[m][n] = np.load(matrix[m][n])
        if m!=n:
            res = res[:m*block_size,:n*block_size]
        return res
    else:
        m = len(matrix)
        res = np.zeros(shape=(m * block_size, m*block_size))
        for i in range(m):
            # res[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size] = np.load(matrix[i])
            res[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size] = scipy.sparse.load_npz(matrix[i]).todense()
                # res[m][n] = np.load(matrix[m][n])
        return res


def output_PDAAQs(P, Ts, U, DQ):
    tmp1 = diag_block_add(U, Ts, "tmp1")
    tmp2 = diag_block_dot(tmp1, P, "tmp2", dot_side="left")
    return diag_block_dot(tmp2, DQ, "PDAAQs", dot_side="right")



def path_sampling(A, T, m, m_):
    """
    display path sampling algorithm
    :param A: sparse matrix of the graph
    :param T: window size of deepwalk
    :param m: non-zero number of the
    :return: sparsed graph after path sampling (dok_matrix)
    """
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
            current_node = random.sample(adjacency_dict[start_node], 1)[0]
            walk_path.append(current_node)
            # update
            start_node = current_node
        return current_node, walk_path

    # if not A.issparse():
    #     print("Matrix A is not sparse matrix")
    # elif not A.isspmatrix_dok():
    #     A = A.todok()

    A_sparse = dok_matrix(A.shape)

    A = dict(A)
    A_paths = list(A.keys())
    adjacency_dict = {}
    for path in A_paths:
        if path[0] not in list(adjacency_dict):
            adjacency_dict[path[0]] = [path[1]]
        adjacency_dict[path[0]].append(path[1])

    for i in range(1, m_+1):
        # random select a edge e(u,v) from origin graph G
        # the selected edge is from u to v
        u = random.sample(list(adjacency_dict.keys()), 1)[0]
        v = random.sample(adjacency_dict[u], 1)[0]

        # random select two integer number
        r = random.sample(list(range(1, T+1)), 1)[0]
        k = random.sample(list(range(1, r+1)), 1)[0]

        u0, u_path = walk(adjacency_dict, u, k-1)
        ur, v_path = walk(adjacency_dict, v, r-k)

        # u_path.reverse()
        # p = u_path + v_path
        p = [(u_path[i], u_path[i+1]) for i in range(len(u_path)-1)] + [(u, v)] + [(v_path[i], v_path[i+1]) for i in range(len(v_path)-1)]
        # print(p, u_path, v_path, u, v)
        r = len(p)

        Zp = 0
        ss = 0
        for path_node_index in range(len(p)):
            # path_node_pairs = (p[path_node_index], p[path_node_index+1])
            path_node_pairs = p[path_node_index]
            path_weight = A[path_node_pairs]
            ss+=1
            # calculate Zp
            Zp += 1/path_weight

        sparse_weight = (2*r*m)/m_*Zp
        A_sparse[u0, ur] = sparse_weight
    # collect garbage
    gc.collect()
    return A_sparse

            




































def path_sampling_block(A, block_size, name):
    """
    block sampling is not necessary
    :param A:
    :param block_size:
    :param name:
    :return:
    """
    a = len(A)
    Final = []
    for i in range(0, a):
        Final.append([])
        for j in range(0, a):
            # b1 = np.load(A[i][j])
            block = scipy.sparse.load_npz(A[i][j]).todense()
            matrix_name = "matrices/" + name + "_" + str(i) + '_' + str(j) + '_' + str(
                block_size) + ".npz"
            # np.save(matrix_name, sum)
            scipy.sparse.save_npz(matrix_name, scipy.sparse.csc_matrix(sum))
            Final[i].append(matrix_name)
    return Final


def test_block_dot():
    test_matrix = random_matrix(1000)
    block_size = 50
    mult_res = np.dot(test_matrix, test_matrix)
    test_matrix = block_matrix_splitter(test_matrix, "test_matrix", block_size)
    block_res = show_block_matrix(block_dot(test_matrix, test_matrix, block_size, "block_res"), block_size)
    print(block_res)
    print("done")

def test_mask():
    test_m = random_matrix(100)
    block_size = 10
    test_matrix = block_matrix_splitter(test_m, "test_matrix", block_size)
    mask_matrix_block = block_mask(test_matrix, block_size,"mask_matrix","left")
    masked = show_block_matrix(test_matrix, block_size)
    block_demask(test_matrix,mask_matrix_block,"left")
    demasked = show_block_matrix(test_matrix, block_size)
    print()



if __name__ == '__main__':
    test_mask()
    # test_block_dot()
    # a = np.random.randint(2, 10, size=(10, 10))
    # b = np.random.randint(2, 10, size=(10, 10))
    # dotres = np.dot(a, b)
    # res = a@b
    # matrixres = np.array(matrixMul(a, b))
    # print(dotres)
    # print(res)
    # print(matrixres)
    # print("111")
