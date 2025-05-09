
import csv
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if __name__ == '__main__':
    # 加载数据
    df_miRNA_drug = pd.read_csv('../../../dataset/miRNA_mRNA.csv', header=None)
    miRNA_drug_matrix = df_miRNA_mRNA.values
    print(miRNA_mRNA_matrix.shape)

    # 创建一个空图
    G = nx.Graph()

    # 添加miRNA节点
    for i in range(1041):
        G.add_node(f"miRNA_{i}")

    # 添加mRNA节点
    for i in range(130):
        G.add_node(f"drug_{i}")

    # 遍历矩阵，添加边
    for miRNA_index in range(1041):
        for mRNA_index in range(130):
            interaction = miRNA_drug_matrix[miRNA_index, drug_index]
            if interaction != 0:
                G.add_edge(f"miRNA_{miRNA_index}", f"drug_{drug_index}")

    # 创建 Node2Vec 对象
    node2vec = Node2Vec(G, dimensions=128, walk_length=150, num_walks=200, workers=1)

    # 训练模型
    model = node2vec.fit()

    # 提取 miRNA 特征向量
    miRNA_features = {}
    for miRNA_node in range(1041):
        miRNA_node_str = f"miRNA_{miRNA_node}"
        miRNA_features[miRNA_node_str] = model.wv[miRNA_node_str]

    df = pd.DataFrame.from_dict(miRNA_features, orient='index')

    # 重命名 DataFrame 的列名，将每一维作为一列
    df.columns = [f'Dimension_{i}' for i in range(128)]  # 列名自定义
    df.to_csv('../../../feature/miRNA_drug_feature_128.csv')
