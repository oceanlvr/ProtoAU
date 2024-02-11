import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp

class Interaction(Data,Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self,conf,training,test)
        # 这里的key是自己根据读进数据建立的 value(其实是数据集给的id) 是数据集的key
        self.user = {} # user2id 这里的id是矩阵的id
        self.item = {}
        self.id2user = {} # id2user
        self.id2item = {}
        # defaultdict 和 dict 相比，如果training_set_u[x]是一个空，则会返回dict {}
        self.training_set_u = defaultdict(dict) # training_set_u 是一个二维数组 training_set_u[u][i] 是 score
        self.training_set_i = defaultdict(dict) # training_set_i 是一个二维数组 training_set_i[i][u] 是 score
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        # A 邻接矩阵
        self.ui_adj = self.__create_sparse_bipartite_adjacency() # 这个是一个UI矩阵4个区域对角上是0，其余两部分是u\i组成的node之间的interaction。
        # A 的拉普拉斯矩阵 D^(-1/2)*A*D^(-1/2) 邻接矩阵的正则化矩阵
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        # popularity_user = {}
        # for u in self.user:
        #     popularity_user[self.user[u]] = len(self.training_set_u[u])
        # popularity_item = {}
        # for u in self.item:
        #     popularity_item[self.item[u]] = len(self.training_set_i[u])

    # 这里是填充 training_set_u/training_set_i user/id2user test_set/test_set_item 部分
    def __generate_set(self):
        # training_data 是一个 [u,i,score]的三元组，其中u是用户，i是物品，score是打分或者是客户的rating
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        for entry in self.test_data: # 和 training_data 数据结构一致
            user, item, rating = entry
            if user not in self.user:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)

    # 构造一个UI矩阵 看公式Neural Graph Collaborative Filtering(8)
    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data] # user
        col_idx = [self.item[pair[1]] for pair in self.training_data] # item
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32) # 全一的数组，因为这个任务里面没有考虑评分 所以都是1
        # Compressed Sparse Row matrix -> csr matrix
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
        # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        #                               这里加了user_num，也就是说从item是从user_num编号后一个开始算起的。
        # row = np.array([0, 0, 1, 2, 2, 2])
        # col = np.array([0, 2, 2, 0, 1, 2])
        # data = np.array([1, 2, 3, 4, 5, 6])
        # csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
        # array([[1, 0, 2],
        #     [0, 0, 3],
        #     [4, 5, 6]])
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T             # 这步骤是在构造无向图
        if self_connection:                       # 如果是自连接则需要在对角上加上1.
            adj_mat += sp.eye(n_nodes)            # 邻接矩阵最后的形式
        return adj_mat

    # UI矩阵（需要保持正确的下标），UI矩阵转为A矩阵，即公式8。最后正则化
    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix(
            (ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),
            shape=(n_nodes, n_nodes),
            dtype=np.float32
        )
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m
