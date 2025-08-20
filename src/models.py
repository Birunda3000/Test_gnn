# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv,
    APPNP, SGConv
)

class GCNNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden)
        self.conv2 = GCNConv(hidden, out_c)
    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, ei)

class SAGENet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_c)
    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, ei)

class GATNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c, heads=4):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GATConv(embedding_dim, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, out_c, heads=1)
    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, ei)

class GINNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn1 = torch.nn.Sequential(torch.nn.Linear(embedding_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden))
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, out_c))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, ei)

class APPNPNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c, K=10, alpha=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.lin1 = torch.nn.Linear(embedding_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, out_c)
        self.prop = APPNP(K=K, alpha=alpha)
    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return self.prop(x, ei)

class SGConvNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c, K=2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv = SGConv(embedding_dim, out_c, K=K)
    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        return self.conv(x, ei)

# Mapeia o nome do modelo para a classe, para facilitar a chamada
ALL_MODELS = {
    "GCNNet": GCNNet,
    "SAGENet": SAGENet,
    "GATNet": GATNet,
    "GINNet": GINNet,
    "APPNPNet": APPNPNet,
    "SGConvNet": SGConvNet,
}