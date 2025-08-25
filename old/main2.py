import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv,
    APPNP, SGConv
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =======================
# Modelos GNN Otimizados com Camada de Embedding
# =======================
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
        # SGConv é um modelo linear, então não precisamos de uma camada oculta extra aqui
        self.conv = SGConv(embedding_dim, out_c, K=K)

    def forward(self, x_indices, ei):
        x = self.embedding(x_indices)
        return self.conv(x, ei)

# =======================
# Funções para carregar dados
# =======================
def load_data(edges_path, target_path, target_col, id_col='id'):
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
    y = torch.full((num_nodes,), -1, dtype=torch.long) # Usar -1 como default para nós sem label
    for _, row in target_df.iterrows():
        if row[id_col] in id_map:
            y[id_map[row[id_col]]] = cat_map[row[target_col]]
    
    # [MUDANÇA CRÍTICA] Em vez da matriz identidade, passamos os índices dos nós.
    # A camada de embedding usará isso para buscar as features.
    x = torch.arange(num_nodes)

    # Máscaras de treino/validação/teste
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

# =======================
# Treinamento e Avaliação
# =======================
def train(model, data, epochs=1000, verbose=True):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-5)
    for epoch in range(epochs):
        opt.zero_grad()
        # [MUDANÇA] Passa os índices (data.x) para o modelo
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        if verbose and (epoch % 10 == 0 or epoch == epochs-1):
            with torch.no_grad():
                pred = out[data.train_mask].argmax(dim=1)
                acc = (pred == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")
    return model

def get_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1

def test(model, data):
    model.eval()
    with torch.no_grad():
        # [MUDANÇA] Passa os índices (data.x) para o modelo
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
    
    results = {}
    for name, mask in zip(['train', 'val', 'test'], [data.train_mask, data.val_mask, data.test_mask]):
        if mask.sum() > 0:
            y_true = data.y[mask].cpu().numpy()
            y_pred = preds[mask].cpu().numpy()
            acc, prec, rec, f1 = get_metrics(y_true, y_pred)
            results[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
    return results

# =======================
# Execução Principal
# =======================
datasets_info = {
    "Musae-Facebook": {
        "loader": lambda: load_data(
            "data/musae-facebook/facebook_large/musae_facebook_edges.csv",
            "data/musae-facebook/facebook_large/musae_facebook_target.csv",
            target_col='page_type'
        )
    },
    "Musae-Github": {
        "loader": lambda: load_data(
            "data/musae-github/git_web_ml/musae_git_edges.csv",
            "data/musae-github/git_web_ml/musae_git_target.csv",
            target_col='ml_target'
        )
    },
}

# [MUDANÇA] Hiperparâmetro para a dimensão das features aprendidas
EMBEDDING_DIM = 512
HIDDEN_DIM = 128
results_table = []

# Descomente a linha do Facebook se quiser rodar para ele também
# dataset_to_run = ["Musae-Facebook", "Musae-Github"]
dataset_to_run = ["Musae-Github"] 

for name in dataset_to_run:
    try:
        info = datasets_info[name]
        data = info["loader"]()
        
        print(f"\n=== Dataset: {name} ===")
        print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
        print(f"Features: {EMBEDDING_DIM} (aprendidas), Classes: {int(data.y.max().item()) + 1}")

        out_c = int(data.y.max().item()) + 1

        for ModelClass in [GCNNet, SAGENet, GATNet, GINNet, APPNPNet, SGConvNet]:
            print(f"\nTreinando {ModelClass.__name__}...")
            
            # [MUDANÇA] Instanciação do modelo agora requer num_nodes e embedding_dim
            if ModelClass is GATNet:
                model = ModelClass(num_nodes=data.num_nodes, embedding_dim=EMBEDDING_DIM, hidden=HIDDEN_DIM, out_c=out_c, heads=4)
            else:
                model = ModelClass(num_nodes=data.num_nodes, embedding_dim=EMBEDDING_DIM, hidden=HIDDEN_DIM, out_c=out_c)
            
            model = train(model, data, epochs=10000, verbose=True)
            metrics = test(model, data)
            
            results_table.append({
                'Dataset': name, 'Model': ModelClass.__name__,
                'Test Acc': metrics['test']['acc'], 'Test F1': metrics['test']['f1'],
                'Val Acc': metrics['val']['acc'], 'Val F1': metrics['val']['f1'],
            })
            print(f"Resultados para {ModelClass.__name__}: Test Acc: {metrics['test']['acc']:.4f}, Test F1: {metrics['test']['f1']:.4f}")

    except Exception as e:
        print(f"[ERRO ao processar {name}]: {e}")

# Printar tabela de resultados
print("\n==== Resultados Finais ====")
df_results = pd.DataFrame(results_table)
print(df_results.to_string(index=False, float_format="%.4f"))