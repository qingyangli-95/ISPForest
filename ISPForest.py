"""
@Qingyang Li
Streaming Half Space Trees for anomaly detection
"""
import numpy as np
import matplotlib.pyplot as plt
from utilities import IncrementalVar, entropy_freqs, Batch
# import pandas as pd
# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import MinMaxScaler
# from threading import Thread

EPSILON = np.finfo(np.float32).eps


class Node:
    def __init__(self, node_id, level, parent=None,
                 inst_ref=0, inst_lat=0, is_leaf=False,
                 norm_ref=0, norm_lat=0, abno_ref=0,
                 abno_lat=0):
        self.id = node_id
        self.level = level
        self.is_leaf = is_leaf
        self.feature = None
        self.split = None
        self.is_terminal = False
        self.inst_ref = inst_ref
        self.inst_lat = inst_lat
        self.norm_ref = norm_ref
        self.norm_lat = norm_lat
        self.abno_ref = abno_ref
        self.abno_lat = abno_lat
        self.parent = parent
        self.left = None
        self.right = None

    @property
    def normal(self):
        return self.norm_ref + self.norm_lat

    @property
    def abnormal(self):
        return self.abno_ref + self.abno_lat

    def update(self, adaptive=0):
        if adaptive < 0 or adaptive > 1:
            raise ValueError('adaptive must in [0, 1]')
        self.inst_ref = adaptive * self.inst_ref + (1 - adaptive) * self.inst_lat
        self.inst_lat = 0
        self.norm_ref = adaptive * self.norm_ref + (1 - adaptive) * self.norm_lat
        self.norm_lat = 0
        self.abno_ref = adaptive * self.abno_ref + (1 - adaptive) * self.abno_lat
        self.abno_lat = 0

    def __repr__(self):
        if self.is_terminal:
            return "%d *\n%d, %.3f\n%.1f, %d\n%.1f, %d\n%.1f, %d" % (self.id,
                                                                     self.feature if self.feature else 0,
                                                                     self.split if self.split else 0,
                                                                     self.inst_ref, self.inst_lat, self.norm_ref,
                                                                     self.norm_lat, self.abno_ref, self.abno_lat)
        else:
            return "%d\n%d, %.3f\n%.1f, %d\n%.1f, %d\n%.1f, %d" % (self.id,
                                                                   self.feature if self.feature else 0,
                                                                   self.split if self.split else 0,
                                                                   self.inst_ref, self.inst_lat, self.norm_ref,
                                                                   self.norm_lat, self.abno_ref, self.abno_lat)


class StreamingHalfSpaceTree:
    def __init__(self, n_trees=25,
                 max_depth=20,
                 min_depth=3,  # ll
                 terminal_depth=7,
                 window_size=250,
                 contamination=0.1,
                 adaptive=0.95):  # ll
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.terminal_depth = min(terminal_depth, max_depth - 1)
        self.min_depth = min(min_depth, max_depth)
        self.window_size = window_size
        self.contamination = contamination
        self.adaptive = adaptive
        self.mass_mv = [IncrementalVar() for _ in range(n_trees)]  # ll
        # 每一棵树 mass 的均值和方差
        self.terminal_entropys = []  # ll
        # 每棵树叶节点inst_ref的熵
        self.__n_nodes = 0
        self.__ins_count = 0

    def set_contamination(self, contamination):
        self.contamination = contamination
        self.threshold = np.sort(self.s_scores)[int(self.window_size * self.contamination)]

    @property
    def node_id(self):
        self.__n_nodes += 1
        return self.__n_nodes

    def init_work_space(self, ndims):
        sqs = np.random.uniform(size=ndims)
        work_range = 2 * np.maximum(sqs, 1 - sqs)
        # work_range = 1.5 * np.maximum(sqs, 1-sqs)
        maxqs = sqs + work_range
        minqs = sqs - work_range
        return sqs, minqs, maxqs

    def choose_dim_by_len(self, mins, maxs):
        lens = np.abs(maxs - mins)
        plens = lens / np.sum(lens)
        feature = np.random.choice(len(mins), p=plens)
        return feature

    def tree_growth(self, node, mins, maxs): # 树的构造
        if node.level == self.terminal_depth:
            node.is_terminal = True
        if node.level >= self.max_depth:
            node.is_leaf = True
            return
        # feature = np.random.choice(len(mins))
        # split = (mins[feature] + maxs[feature])/2
        feature = self.choose_dim_by_len(mins, maxs)  # ll
        split = np.random.uniform(low=mins[feature], high=maxs[feature])
        node.feature = feature
        node.split = split
        newmins = mins.copy()
        newmins[feature] = split
        newmaxs = maxs.copy()
        newmaxs[feature] = split
        node.left = Node(self.node_id, node.level + 1)
        node.left.parent = node
        node.right = Node(self.node_id, node.level + 1)
        node.right.parent = node
        self.tree_growth(node.left, mins, newmaxs)
        self.tree_growth(node.right, newmins, maxs)

    def traverse_record_terminals(self, node, leaves):
        if node.is_terminal:
            leaves.append(node)
            return
        self.traverse_record_terminals(node.left, leaves)
        self.traverse_record_terminals(node.right, leaves)

    def calc_ternimal_entropy(self):  ## 改进
        self.ternimal_entropys = []
        for root in self.trees:
            ternimal_list = []
            self.traverse_record_terminals(root, ternimal_list)  ##
            entropy = entropy_freqs([al.inst_ref * (2 ** al.level) for al in ternimal_list])
            # self.ternimal_entropys.append(1 - entropy / self.max_depth)
            self.ternimal_entropys.append(1 - entropy / self.max_depth)
        z = sum(self.ternimal_entropys)
        self.ternimal_entropys = [aen / z for aen in self.ternimal_entropys]
        return

    def traverse(self, node, x,
                 update_type='inst_lat',
                 update=True,
                 return_node=None):
        """
        @update_type: 'inst_ref', 'inst_lat', 'norm_feed', 'abno_feed'
        """
        if update and update_type == 'inst_ref':
            node.inst_ref += 1
        if update and update_type == 'inst_lat':
            node.inst_lat += 1

        if update and update_type == 'norm_feed':
            node.norm_lat += 1
        if update and update_type == 'abno_feed':
            node.abno_lat += 1
            node.inst_lat = max(0, node.inst_lat - 1)

        if node.is_terminal:
            return_node = node
        if node.is_leaf:
            if return_node:
                node = return_node
            return node
        feature = node.feature
        if x[feature] < node.split:
            return self.traverse(node.left, x, update_type, update, return_node)
        else:
            return self.traverse(node.right, x, update_type, update, return_node)

    def mass_on_nodes(self, nodes):
        masses = []
        for k, anode in enumerate(nodes):
            mass = anode.inst_ref * (2 ** anode.level)
            mass_mv = self.mass_mv[k]
            mass_mv(mass)
            masses.append(mass)
        return masses

    def score_on_masses(self, masses):  #
        scores = []
        consistencies = []
        for masslist in masses:
            raw_scores = [1 - self.sigmoid(amass, amv)
                          for amass, amv in zip(masslist, self.mass_mv)]
            avg_score = np.mean(raw_scores)
            scores.append(avg_score)
        return scores

    def score_on_nodes(self, nodes, return_score_list=False,
                       update_mass=True):
        scores = []
        errors = []
        for k, anode in enumerate(nodes):
            mass = anode.inst_ref * (2 ** anode.level)
            mass_mv = self.mass_mv[k]
            if update_mass:
                mass_mv(mass)
            smi = self.sigmoid(mass, mass_mv)
            scores.append(1 - smi)
            aini = anode.normal + anode.abnormal
            if aini > 0:
                expected = anode.normal / aini
            else:
                expected = smi
            error = np.abs(expected - smi)
            errors.append(error)
        # consistency = entropy_freqs(scores)/np.log2(len(self.trees))
        # consistency = np.std(scores)
        consistency = np.mean(errors)
        scores = np.array(scores)
        avg_score = np.mean(scores)
        # terminal inst_ref 的diversity，是否有冲突?
        # raw_score = np.average(masses, weights=self.ternimal_entropys)
        # 论文
        if return_score_list:
            return avg_score, consistency, scores
        return avg_score, consistency

    def sigmoid(self, x, mass_mv):  # 论文
        if mass_mv.var < 1e-10:
            mass_mv(0)
        # gamma = np.sqrt(3 * mass_mv.var) / np.pi
        gamma = np.pi * np.sqrt(mass_mv.var/ 3)
        return 1.0 / (1.0 + np.exp((-x + mass_mv.mean) / gamma))

    # 数据fit到树结构当中
    def fit(self, X):
        X = np.array(X)
        N, M = X.shape
        self.trees = []
        for i in range(self.n_trees):
            _, mins, maxs = self.init_work_space(M)
            aroot = Node(self.node_id, 0)
            self.tree_growth(aroot, mins, maxs)
            for x in X:
                self.traverse(aroot, x, update_type='inst_ref')
            self.trees.append(aroot)
        self.calc_ternimal_entropy()
        masses = np.array([self.mass_on_nodes([self.traverse(atree, x, update=False)
                                               for atree in self.trees]) for x in X])
        self.s_scores = self.score_on_masses(masses)
        self.threshold = np.sort(self.s_scores)[int(N * (1 - self.contamination))]
        return self

    def update_tree(self, node):
        node.update(adaptive=self.adaptive)
        if node.is_leaf:
            return
        self.update_tree(node.left)
        self.update_tree(node.right)

# 根据fit的树来预测
    def predict(self, x, cut=True,
                scale_score = False,
                return_consistency=False):
        self.__ins_count += 1
        if self.__ins_count >= self.window_size:
            for atree in self.trees:
                self.update_tree(atree)
            self.calc_ternimal_entropy()
            self.__ins_count = 0
        terminals = [self.traverse(atree, x, update_type='inst_lat') for atree in self.trees]
        score, consistency = self.score_on_nodes(terminals)
        if return_consistency:
            score = [score, consistency]
        if not cut:
            return score
        else:
            return int(score > self.threshold), consistency

    def expand_node(self, node, t, y, mass_mv):
        if node.is_leaf:
            return False
        gl, rl = self.node_derivative(node.left, t, y, mass_mv)
        gr, rr = self.node_derivative(node.right, t, y, mass_mv)
        if rl < -EPSILON and rr < -EPSILON:
            node.is_terminal = False
            node.left.is_terminal = True
            node.right.is_terminal = True
            return True
        return False

    def collapse_node(self, node, t, y, mass_mv):
        if node.level <= self.min_depth:
            return False
        parent = node.parent
        if node.id == parent.left.id:
            brother = parent.right
        else:
            brother = parent.left
        terminals = []
        self.traverse_record_terminals(brother, terminals)
        derivatives = np.array([self.node_derivative(aterm, t, y, mass_mv)
                                for aterm in terminals])
        rs = derivatives[:, 1]
        if all(rs > EPSILON):
            for aterm in terminals:
                aterm.is_terminal = False
            node.is_terminal = False
            parent.is_terminal = True
            return True
        return False

    def node_derivative(self, node, t, y, mass_mv):
        mass = node.inst_ref * (2 ** node.level)
        smi = self.sigmoid(mass, mass_mv)
        global_de = smi * (1 - smi) * (y - t) / (y * (1 - y) * self.n_trees)
        ni, ai = node.normal, node.abnormal
        local_de = ni - (ai + ni) * smi
        return global_de, local_de

    # def feed_back(self, x, label, adjust_rate=0.8):
    def feed_back(self, x, label):
        if label == 0:
            feed_type = 'norm_feed'
        else:
            feed_type = 'abno_feed'
        terminals = [self.traverse(atree, x, update_type=feed_type) for atree in self.trees]
        t = int(label)
        y, consistency, s = self.score_on_nodes(terminals, return_score_list=True, update_mass=False)
        global_derivative = s * (1 - s) * (y - t) / (y * (1 - y) * len(terminals))
        nas = np.array([[aterminal.normal, aterminal.abnormal] for aterminal in terminals])
        ni = nas[:, 0]
        ai = nas[:, 1]
        regional_derivative = ni - (ai + ni) * s

        for gi, ri, node, amv in zip(global_derivative,
                                     regional_derivative,
                                     terminals, self.mass_mv):
            derivative_sign = gi * ri
            adjust_rate = gi/ri  # 自适应的adjust_rate
            if derivative_sign > EPSILON:  # g and r have the same sign
                if ri > EPSILON:  # g>0 and r>0: collapse or increase mass
                    trycollapse = self.collapse_node(node, t, y, amv)
                    if not trycollapse:
                        node.inst_ref += adjust_rate * (gi + ri) * (2 ** node.level)
                        # node.inst_ref += adjust_rate * (gi + ri)
                else:  # # g<0 and r<0: expand or decrease mass
                    tryexpand = self.expand_node(node, t, y, amv)
                    if not tryexpand:
                        node.inst_ref += adjust_rate * (ri + gi) * (2 ** node.level)
                        # node.inst_ref += adjust_rate * (ri + gi)
            elif derivative_sign < -EPSILON:  # g,r conflicts, expand the terminal if possible
                self.expand_node(node, t, y, amv)
                # tryexpand = self.expand_node(node, t, y, amv)
                # if not tryexpand:
                #     node.inst_ref += adjust_rate * abs(ri+gi) * (2 ** node.level)
        return


# def traverse(node, edges, max_depth, terminal=False):
#     if terminal and node.is_terminal:
#         return
#     if node.is_leaf:
#         return
#     if max_depth and node.level >= max_depth:
#         return
#     edges.append([str(node), str(node.left)])
#     edges.append([str(node), str(node.right)])
#     traverse(node.left, edges, max_depth, terminal)
#     traverse(node.right, edges, max_depth, terminal)
#
#
# def traverse_record_leaves(node, leaves):
#     if node.is_leaf:
#         leaves.append(node)
#         return
#     traverse_record_leaves(node.left, leaves)
#     traverse_record_leaves(node.right, leaves)
#
#
# def tree_leaf_entropy(root):
#     leaf_list = []
#     traverse_record_leaves(root, leaf_list)
#     leaf_number = [al.inst_ref for al in leaf_list]
#     return entropy_freqs(leaf_number)


# def draw_edges(edges, graph_name="tmp", file_name="tmp"):
#     from graphviz import Digraph
#     u = Digraph(graph_name, file_name)
#     for x, y in edges:
#         u.edge(x, y)
#     u.view()
#     return u
#
# def tree_vis(root, max_depth=None, terminal=False):
#     edges = []
#     traverse(root, edges, max_depth, terminal)
#     draw_edges(edges)
#     return
#
# def test_init_work_space(points=100):
#     hst = StreamingHalfSpaceTree()
#     sq, mins, maxs = hst.init_work_space(points)
#     plt.scatter(sq, mins)
#     plt.scatter(sq, maxs)
#     plt.show()
#
#
# def synthetic_anomaly(norm_ref=100, abno_ref = 20):
#     rng = np.random.RandomState(42)
#     # 正常数据
#     X = 0.3 * rng.randn(norm_ref, 2)
#     X_train = np.r_[X + 2, X - 2]
#     # 异常数据
#     X_outliers = rng.uniform(low=-4, high=4, size=(abno_ref, 2))
#     X = np.concatenate([X_train, X_outliers])
#     y = np.array(2*[1]*norm_ref + [-1]*abno_ref)
#     idx  = list(range(2*norm_ref+abno_ref))
#     from random import shuffle
#     shuffle(idx)
#     return X[idx], y[idx]
#
#
# def synthetic_demo():
#     from utilities import Batch
#     from time import sleep
#     x, y = synthetic_anomaly(norm_ref=880, abno_ref=120)
#     x = x/np.max(x, axis=0).reshape([1, -1])
#     batch = Batch(x, y)
#     trainn = 250
#     plt.scatter(x[:trainn, 0], x[:trainn, 1])
#     plt.show()
#
#     hst = StreamingHalfSpaceTree(window_size=trainn,
#                     n_trees=15,
#                     max_depth=5,
#                     terminal_depth=2,
#         contamination=0.1, adaptive=0.5).fit(batch.next(trainn)[0])
#     tree_vis(hst.trees[0], max_depth=3)
#     while batch.epochs <=1:
#         data = batch.next(trainn)[0]
#         for i,d in enumerate(data):
#             score = hst.predict(d)
#             plt.scatter(d[0], d[1], color = "red" if score==1 else "green")
#             #if (i+1) % 50 ==0:
#             #    tree_vis(hst.trees[0], max_depth=3)
#             #    sleep(1)
#         # for i in range(5):
#             # splits = tree_vis(hst.trees[i], max_depth=3)
#             # plt.vlines(splits[0], -1, 1, color='blue')
#             # plt.hlines(splits[1], -1, 2, color='green')
#             # plt.show()
#
#
# if __name__ == "__main__":
#     synthetic_demo()