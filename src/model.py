# model.py

"""
Define as arquiteturas dos modelos GNN.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, APPNP, SGConv
)

# Modelo original para referência
class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x.to_dense(), edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

# --- Seus novos modelos adaptados ---

class GCNNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hidden)
        self.conv2 = GCNConv(hidden, out_c)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x.to_dense(), edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

class SAGENet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c):
        super().__init__()
        self.conv1 = SAGEConv(in_c, hidden)
        self.conv2 = SAGEConv(hidden, out_c)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x.to_dense(), edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

class GATNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_c, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, out_c, heads=1)
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x.to_dense(), edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

class GINNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c):
        super().__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(in_c, hidden),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hidden, hidden))
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden, hidden),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hidden, out_c))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x.to_dense(), edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

class APPNPNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_c, hidden)
        self.lin2 = torch.nn.Linear(hidden, out_c)
        self.prop = APPNP(K=K, alpha=alpha)
    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x.to_dense()))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return torch.sigmoid(x)

class SGConvNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c, K=2):
        super().__init__()
        self.conv = SGConv(in_c, out_c, K=K)
    def forward(self, x, edge_index):
        x = self.conv(x.to_dense(), edge_index)
        return torch.sigmoid(x)