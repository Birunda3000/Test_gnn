# src/data_loader.py

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def load_data(edges_path, target_path, target_col, id_col='id'):
    """Carrega os dados de um dataset, processa e retorna um objeto Data do PyG."""
    edges_df = pd.read_csv(edges_path)
    target_df = pd.read_csv(target_path)

    node_ids = np.sort(np.unique(np.concatenate([edges_df['id_1'], edges_df['id_2']])))
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    edge_index = np.array([
        [id_map[src], id_map[dst]]
        for src, dst in zip(edges_df['id_1'], edges_df['id_2']) if src in id_map and dst in id_map
    ]).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    categories = sorted(target_df[target_col].unique())
    cat_map = {cat: i for i, cat in enumerate(categories)}
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for _, row in target_df.iterrows():
        if row[id_col] in id_map:
            y[id_map[row[id_col]]] = cat_map[row[target_col]]

    x = torch.arange(num_nodes)

    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    train_ratio, val_ratio = 0.7, 0.15
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[idx[:train_end]] = True
    val_mask[idx[train_end:val_end]] = True
    test_mask[idx[val_end:]] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data