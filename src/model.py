# model.py

"""
Define a arquitetura do modelo GNN.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # A conversão para denso pode ser necessária para a primeira camada em algumas versões do PyG
        # ao lidar com tensores esparsos.
        x = self.conv1(x.to_dense(), edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)