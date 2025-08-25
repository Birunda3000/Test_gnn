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
from src import config
import numpy as np

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
    with open(config.FEATURES_FILE, 'r') as f:
        features_data = json.load(f)
    node_ids = sorted(features_data.keys(), key=int)
    feature_lists = [features_data[node_id] for node_id in node_ids]
    mlb = MultiLabelBinarizer(sparse_output=True)
    X_features_scipy = mlb.fit_transform(feature_lists)
    
    coo = X_features_scipy.tocoo()

    # --- CORREÇÃO 1 APLICADA AQUI ---
    # Juntamos os arrays numpy primeiro para maior eficiência
    indices_np = np.vstack((coo.row, coo.col))
    feature_indices = torch.LongTensor(indices_np)
    
    feature_values = torch.FloatTensor(coo.data)

    # --- CORREÇÃO 2 APLICADA AQUI ---
    # Usamos a nova função recomendada para criar o tensor esparso
    feature_tensor_sparse = torch.sparse_coo_tensor(
        feature_indices, feature_values, coo.shape, dtype=torch.float
    )

    # Carregar Arestas
    edges_df = pd.read_csv(config.EDGES_FILE)
    edge_index = torch.LongTensor(edges_df.to_numpy().T)

    # Carregar Rótulos
    target_df = pd.read_csv(config.TARGET_FILE)
    target_df = target_df.sort_values('id').reset_index(drop=True)
    labels = torch.FloatTensor(target_df['ml_target'].values).unsqueeze(1)
    
    return feature_tensor_sparse, edge_index, labels

def _create_masks(data):
    """Cria máscaras de treino, validação e teste."""
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_size = int(num_nodes * config.TRAIN_RATIO)
    val_size = int(num_nodes * config.VALIDATION_RATIO)
    
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