# data_loader.py

"""
Responsável por carregar, processar e criar o objeto de dados do grafo,
incluindo as máscaras de treino/validação/teste.
"""

import json
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data

def load_and_prepare_data():
    """
    Função principal que carrega todos os arquivos, processa os dados,
    cria as máscaras e retorna um objeto Data finalizado.
    """
    print("Iniciando carregamento e preparação dos dados...")

    # Carregar features, arestas e rótulos
    X_features, edge_index, labels = _load_raw_data()
    
    # Criar o objeto Data
    graph_data = Data(x=X_features, edge_index=edge_index, y=labels)
    
    # Criar e adicionar as máscaras
    graph_data = _create_masks(graph_data)

    print("Dados carregados e preparados com sucesso.")
    return graph_data

def _load_raw_data():
    """Função auxiliar para carregar e converter os dados brutos."""
    # Processar Features
    with open('musae_git_features.json', 'r') as f:
        features_data = json.load(f)
    node_ids = sorted(features_data.keys(), key=int)
    feature_lists = [features_data[node_id] for node_id in node_ids]
    mlb = MultiLabelBinarizer(sparse_output=True)
    X_features_scipy = mlb.fit_transform(feature_lists)
    
    coo = X_features_scipy.tocoo()
    feature_indices = torch.LongTensor([coo.row, coo.col])
    feature_values = torch.FloatTensor(coo.data)
    feature_tensor_sparse = torch.sparse.FloatTensor(feature_indices, feature_values, coo.shape)

    # Carregar Arestas
    edges_df = pd.read_csv('musae_git_edges.csv')
    edge_index = torch.LongTensor(edges_df.to_numpy().T)

    # Carregar Rótulos
    target_df = pd.read_csv('musae_git_target.csv')
    target_df = target_df.sort_values('id').reset_index(drop=True)
    labels = torch.FloatTensor(target_df['ml_target'].values).unsqueeze(1)
    
    return feature_tensor_sparse, edge_index, labels

def _create_masks(data):
    """Cria máscaras de treino, validação e teste."""
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True
    
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[val_indices] = True

    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_indices] = True
    
    return data