import numpy as np


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1

        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
######################### à modifier - version 1##################################################
        elif layout == 'test1':
            self.num_node = 43
            self_link = [(i, i) for i in range(self.num_node)]

            neighbor_1base = [(3, 4), (4, 5), (3, 5), #central triangle
               (24, 25), (41, 42), # left wrist, right wrist
               (14, 24), (14, 25), (31, 41), (31, 42), # left forearm, right forearm
               (12, 14), (29, 31), # left elbow, right elbow
               (12, 23), (29, 40), # left upper arm, right upper arm
               (23, 20), (40, 37), # left shoulder, right shoulder
               (10, 27), (13, 30), (10, 13), (27, 30), # head
               (6, 8), (6, 7), (7, 8), # trunk
               (0, 1), (0, 2), (1, 2), # back
               (0, 6), (8, 2), (1, 7), # trunk - back
               (8, 37), (7, 20), #trunk - shoulders
               (2, 37), (1, 20), #back - shoulders
               (30, 37), (13, 20), #head - shoulders
               #(2, 20), (7, 20), (6, 37), (0, 37),
               (22, 18), (39, 35), # left toe, right toe
               (18, 16), (22, 16), (39, 33), (35, 33), # left heel, right heel
               (18, 9), (22, 9), (16, 9), (39, 26), (35, 26), (33, 26), # left ankle, right ankle
               (9, 19), (26, 36), # left shank, right shank
               (19, 17), (17, 9), (36, 34), (34, 26), # left knee, right knee
               (17, 21), (34, 38), # left thigh, right thigh
               (15, 11), (15, 32), (32, 28), (28, 11), #Rectangle hip
               (38, 32), (21, 11), (38, 28), (21, 15), #Hips - knee
               (3,28), (5,32), (4,11), (5,15),
               (6,3), (7, 4), (0,5)
              ]

            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 3

######################### à modifier version 2#########################################
        elif layout == 'test':
            self.num_node = 33
            self_link = [(i, i) for i in range(self.num_node)]

            neighbor_1base = [(16, 17), (31, 32), # left wrist, right wrist
               (6, 16), (6, 17), (21, 31), (21, 32), # left forearm, right forearm
               (5, 6), (20, 21), # left elbow, right elbow
               (5, 15), (20, 30), # left upper arm, right upper arm
               (15, 12), (30, 27), # left shoulder, right shoulder
               (4, 19),  # head
               (0, 2), # trunk - back
               (2, 27), (2, 12), #trunk - shoulders
               (0, 27), (0, 12), #back - shoulders
               (19, 27), (4, 12), #head - shoulders
               #(2, 20), (7, 20), (6, 37), (0, 37),
               (14, 10), (29, 25), # left toe, right toe
               (10, 8), (14, 8), (29, 23), (25, 23), # left heel, right heel
               (10, 3), (14, 3), (8, 3), (29, 18), (25, 18), (23, 18), # left ankle, right ankle
               (3, 11), (18, 26), # left shank, right shank
               (11, 9), (9, 3), (26, 24), (24, 18), # left knee, right knee
               (9, 13), (24, 28), # left thigh, right thigh
               (7, 22), #hip
               (28, 22), (13, 7), #Hips - knee
               (1,22), (1,7),
               (2,1), (0,1)
              ]

            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 1

        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            print("pass strategie")
        else:
            raise ValueError("Do Not Exist This Strategy")

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
