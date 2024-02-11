import numpy as np
import scipy.sparse as sp

class Graph(object):
    def __init__(self):
        pass
    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:    
            d_inv = np.power(rowsum, -0.5).flatten() # 1/sqrt(dgree) d^(-1/2)
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv) # d->D Construct a sparse matrix from diagonals. 从对角线构建一个稀疏矩阵。
            norm_adj_tmp = d_mat_inv.dot(adj_mat) # D^(-1/2)*A
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv) # D^(-1/2)*A*D^(-1/2)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat) # D^(-1/2)*A
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        pass
