import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import argparse
import tensorflow.compat.v1 as tf
from trainer import HGCNTrainer


# 超图生成函数
def generate_hypergraph(adj_matrix, threshold=0.8):
    """
    基于邻接矩阵生成超图
    :param adj_matrix: 邻接矩阵，表示节点之间的相似性
    :param threshold: 相似性阈值，只有相似性大于该值的节点才会连接在同一超边中
    :return: 超图的邻接矩阵
    """
    adj_matrix[adj_matrix >= threshold] = 1
    adj_matrix[adj_matrix < threshold] = 0

    # 基于阈值化后的矩阵生成超图
    num_nodes = adj_matrix.shape[0]
    hyperedges = []

    for i in range(num_nodes):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        if len(neighbors) > 1:
            hyperedges.append(neighbors)

    return hyperedges


def build_hypergraph(adj_matrix):
    """
    从邻接矩阵构建超图的邻接矩阵
    :param adj_matrix: 邻接矩阵，表示节点之间的相似性
    :return: 超图的邻接矩阵
    """
    hyperedges = generate_hypergraph(adj_matrix)

    # 将超边转换为超图的邻接矩阵
    num_nodes = adj_matrix.shape[0]
    hypergraph_adj = np.zeros((num_nodes, num_nodes))

    for edge in hyperedges:
        for node in edge:
            for other_node in edge:
                if node != other_node:
                    hypergraph_adj[node, other_node] = 1

    hypergraph_adj_sparse = sp.coo_matrix(hypergraph_adj)
    return hypergraph_adj_sparse


# 超图卷积模型训练函数
def get_hgcns_feature(adj, features, epochs, l):
    args = parse_args(epochs=epochs, l=l)
    feature_dim = features.shape[1]
    args.hidden_dims = [feature_dim] + args.hidden_dims

    G, S, R = prepare_hypergraph_data(adj)
    hgcn_trainer = HGCNTrainer(args)
    hgcn_trainer.train_hypergraph_model(G, features, S, R)
    embeddings, attention = hgcn_trainer.infer_hypergraph_embeddings(G, features, S, R)

    tf.reset_default_graph()
    return embeddings


def prepare_hypergraph_data(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    data = adj.tocoo().data
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col


def parse_args(epochs, l):
    parser = argparse.ArgumentParser(description="Run HGCN.")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate. Default is 0.001.')
    parser.add_argument('--n-epochs', default=epochs, type=int,
                        help='Number of epochs')
    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[256, 128],
                        help='Number of dimensions.')
    parser.add_argument('--lambda-', default=l, type=float,
                        help='Parameter controlling the contribution of graph structure reconstruction in the loss function.')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout.')
    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='Gradient clipping')

    return parser.parse_args()


if __name__ == '__main__':
    df_drug = pd.read_csv('../../../feature/miRNA_drug_feature_128.csv', index_col=0)
    df_func = pd.read_csv('../../../dataset/miRNA_func_sim.csv', header=None)

    feature = df_drug.values
    similarity = df_func.values

    # 生成超图邻接矩阵
    network = build_hypergraph(similarity)
    adj, features = network, feature

    # 获取HGCN的嵌入特征
    embeddings = get_hgcns_feature(adj, features, epochs=100, l=1)
    print(embeddings.shape)

    # 保存特征
    file_path = '../../../feature/hgcns_feature_drug_0.8_128_0.01.csv'
    np.savetxt(file_path, embeddings, delimiter=',')
