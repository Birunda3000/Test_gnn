# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d, Sequential, ReLU, Linear
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, APPNP, SGConv
from src import config


# --- MODELO GCN APRIMORADO ---
class GCNNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList()
        self.bns = ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(embedding_dim, out_c))
        else:
            self.convs.append(GCNConv(embedding_dim, hidden))
            self.bns.append(BatchNorm1d(hidden))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden, hidden))
                self.bns.append(BatchNorm1d(hidden))
            self.convs.append(GCNConv(hidden, out_c))

    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        if len(self.convs) == 1:
            return self.convs[0](x, ei)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, ei)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT, training=self.training)
        x = self.convs[-1](x, ei)
        return x


# --- MODELO SAGENET APRIMORADO ---
class SAGENet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList()
        self.bns = ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(embedding_dim, out_c))
        else:
            self.convs.append(SAGEConv(embedding_dim, hidden))
            self.bns.append(BatchNorm1d(hidden))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden, hidden))
                self.bns.append(BatchNorm1d(hidden))
            self.convs.append(SAGEConv(hidden, out_c))

    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        if len(self.convs) == 1:
            return self.convs[0](x, ei)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, ei)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT, training=self.training)
        x = self.convs[-1](x, ei)
        return x


# --- MODELO GATNET APRIMORADO ---
class GATNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c, num_layers, heads):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList()
        self.bns = ModuleList()

        if num_layers == 1:
            # Para GAT, a camada final sempre tem heads=1
            self.convs.append(GATConv(embedding_dim, out_c, heads=1))
        else:
            # Camada de entrada
            self.convs.append(GATConv(embedding_dim, hidden, heads=heads))
            self.bns.append(BatchNorm1d(hidden * heads))

            # Camadas ocultas
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden * heads, hidden, heads=heads))
                self.bns.append(BatchNorm1d(hidden * heads))

            # Camada de saída
            self.convs.append(GATConv(hidden * heads, out_c, heads=1))

    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        if len(self.convs) == 1:
            return self.convs[0](x, ei)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, ei)
            x = self.bns[i](x)
            x = F.elu(x)  # GAT costuma usar ELU
            x = F.dropout(x, p=config.DROPOUT, training=self.training)
        x = self.convs[-1](x, ei)
        return x


# --- MODELO GINNET APRIMORADO ---
class GINNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList()
        self.bns = ModuleList()

        if num_layers == 1:
            mlp = Sequential(Linear(embedding_dim, out_c))
            self.convs.append(GINConv(mlp))
        else:
            # Camada de entrada
            mlp1 = Sequential(
                Linear(embedding_dim, hidden), ReLU(), Linear(hidden, hidden)
            )
            self.convs.append(GINConv(mlp1))
            self.bns.append(BatchNorm1d(hidden))

            # Camadas ocultas
            for _ in range(num_layers - 2):
                mlp = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))
                self.convs.append(GINConv(mlp))
                self.bns.append(BatchNorm1d(hidden))

            # Camada de saída
            mlp_out = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, out_c))
            self.convs.append(GINConv(mlp_out))

    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        if len(self.convs) == 1:
            return self.convs[0](x, ei)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, ei)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=config.DROPOUT, training=self.training)
        x = self.convs[-1](x, ei)
        return x


# --- MODELO APPNPNET CONFIGURÁVEL ---
class APPNPNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden, out_c, K, alpha):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.lin1 = torch.nn.Linear(embedding_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, out_c)
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=config.DROPOUT, training=self.training)
        x = self.lin2(x)
        return self.prop(x, ei)


# --- MODELO SGCONVNET CONFIGURÁVEL ---
class SGConvNet(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, out_c, K):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        # SGConv é um modelo linear, então não usamos camadas ocultas ou normalização
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
