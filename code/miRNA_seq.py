import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

# 1. 加载数据并创建图
df_ms = pd.read_csv('../../dataset/miRNA_seq_sim.csv', header=None)
ms = df_ms.values
G = nx.Graph()

# 添加节点和边
for i in range(1041):
    G.add_node(i)
for i in range(1041):
    for j in range(i + 1, 1041):
        similarity = ms[i, j]
        if similarity > 0:
            G.add_edge(i, j, weight=similarity)

# 2. 使用 Node2Vec 提取节点特征
node2vec = Node2Vec(G, dimensions=64, walk_length=150, num_walks=200, workers=1, weight_key='weight')
model = node2vec.fit()
ms_features = {str(node): model.wv[str(node)] for node in G.nodes()}
ms_features = pd.DataFrame.from_dict(ms_features, orient='index')

# 3. 转换为 PyTorch Geometric 格式
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
edge_weight = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)
x = torch.tensor(ms_features.values, dtype=torch.float32)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

# 4. 定义 GIN 模型
class GINModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, hidden_feats),
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats),
        ))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# 5. 训练模型
model = GINModel(in_feats=64, hidden_feats=64, out_feats=64)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
labels = torch.randint(0, 2, (1041,))

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 6. 保存特征
model.eval()
with torch.no_grad():
    embeddings = model(data)

gin_features = embeddings.numpy()
gin_df = pd.DataFrame(gin_features)
gin_df.columns = [f'Dimension_{i}' for i in range(gin_features.shape[1])]
gin_df.index.name = 'Index'
gin_df.to_csv('../../feature/miRNA_seq_feature_64.csv')
print("GIN 特征已保存到 '../../feature/miRNA_seq_feature_64.csv'")
