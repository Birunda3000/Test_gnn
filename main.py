import os
import torch
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
# Modelos GNN
# =======================
class GCNNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hidden)
        self.conv2 = GCNConv(hidden, out_c)
    def forward(self, x, ei):
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, ei)

class SAGENet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c):
        super().__init__()
        self.conv1 = SAGEConv(in_c, hidden)
        self.conv2 = SAGEConv(hidden, out_c)
    def forward(self, x, ei):
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, ei)

class GATNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_c, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, out_c, heads=1)
    def forward(self, x, ei):
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, ei)

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
    def forward(self, x, ei):
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, ei)

class APPNPNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_c, hidden)
        self.lin2 = torch.nn.Linear(hidden, out_c)
        self.prop = APPNP(K=K, alpha=alpha)
    def forward(self, x, ei):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return self.prop(x, ei)

class SGConvNet(torch.nn.Module):
    def __init__(self, in_c, hidden, out_c, K=2):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_c, hidden)
        self.conv = SGConv(hidden, out_c, K=K)
    def forward(self, x, ei):
        x = F.relu(self.lin1(x))
        return self.conv(x, ei)

# =======================
# Funções para carregar musae-facebook e musae-github
# =======================
def load_musae_facebook(data_dir):
    edges_path = os.path.join(data_dir, "musae_facebook_edges.csv")
    target_path = os.path.join(data_dir, "musae_facebook_target.csv")
    edges_df = pd.read_csv(edges_path)
    target_df = pd.read_csv(target_path)

    node_ids = np.sort(np.unique(np.concatenate([edges_df['id_1'], edges_df['id_2']])))
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    edge_index = np.array([
        [id_map[src], id_map[dst]]
        for src, dst in zip(edges_df['id_1'], edges_df['id_2'])
    ]).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    page_types = sorted(target_df['page_type'].unique())
    page_type_map = {pt: i for i, pt in enumerate(page_types)}
    y = torch.zeros(len(node_ids), dtype=torch.long)
    for _, row in target_df.iterrows():
        y[id_map[row['id']]] = page_type_map[row['page_type']]

    x = torch.eye(len(node_ids), dtype=torch.float)

    num_nodes = len(node_ids)
    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    train_ratio, val_ratio = 0.7, 0.15
    train_idx = torch.tensor(idx[:int(train_ratio*num_nodes)], dtype=torch.long)
    val_idx = torch.tensor(idx[int(train_ratio*num_nodes):int((train_ratio+val_ratio)*num_nodes)], dtype=torch.long)
    test_idx = torch.tensor(idx[int((train_ratio+val_ratio)*num_nodes):], dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return [data]

def load_musae_github(data_dir):
    edges_path = os.path.join(data_dir, "musae_git_edges.csv")
    target_path = os.path.join(data_dir, "musae_git_target.csv")
    edges_df = pd.read_csv(edges_path)
    target_df = pd.read_csv(target_path)

    node_ids = np.sort(np.unique(np.concatenate([edges_df['id_1'], edges_df['id_2']])))
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    edge_index = np.array([
        [id_map[src], id_map[dst]]
        for src, dst in zip(edges_df['id_1'], edges_df['id_2'])
    ]).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # O label é 'category' no musae-github
    categories = sorted(target_df['ml_target'].unique())
    cat_map = {cat: i for i, cat in enumerate(categories)}
    y = torch.zeros(len(node_ids), dtype=torch.long)
    for _, row in target_df.iterrows():
        y[id_map[row['id']]] = cat_map[row['ml_target']]

    x = torch.eye(len(node_ids), dtype=torch.float)

    num_nodes = len(node_ids)
    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    train_ratio, val_ratio = 0.7, 0.15
    train_idx = torch.tensor(idx[:int(train_ratio*num_nodes)], dtype=torch.long)
    val_idx = torch.tensor(idx[int(train_ratio*num_nodes):int((train_ratio+val_ratio)*num_nodes)], dtype=torch.long)
    test_idx = torch.tensor(idx[int((train_ratio+val_ratio)*num_nodes):], dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return [data]

# =======================
# Treinamento e Avaliação
# =======================
def train(model, data, epochs=200, verbose=True):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        if verbose and (epoch % 10 == 0 or epoch == epochs-1):
            pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
            true = data.y[data.train_mask].cpu().numpy()
            acc = accuracy_score(true, pred)
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
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1).cpu().numpy()
    y = data.y.cpu().numpy()
    results = {}
    for name, mask in zip(['train', 'val', 'test'], [data.train_mask, data.val_mask, data.test_mask]):
        y_true = y[mask.cpu().numpy()]
        y_pred = preds[mask.cpu().numpy()]
        acc, prec, rec, f1 = get_metrics(y_true, y_pred)
        results[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
    return results

# =======================
# Executa para musae-facebook e musae-github
# =======================
datasets = {
    # "Musae-Facebook": lambda: load_musae_facebook("data/musae-facebook/facebook_large"),
    "Musae-Github": lambda: load_musae_github("data/musae-github/git_web_ml"),
}

results_table = []

for name, loader in datasets.items():
    try:
        dataset = loader()
        data = dataset[0]
        print(f"\n=== Dataset: {name} ===")
        print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
        print(f"Features: {data.num_node_features}, Classes: {int(data.y.max().item()) + 1}")

        in_c = data.num_node_features
        out_c = int(data.y.max().item()) + 1

        for ModelClass in [GCNNet, SAGENet, GATNet, GINNet, APPNPNet, SGConvNet]:
            print(f"\nTreinando {ModelClass.__name__}...")
            if ModelClass is GATNet:
                model = ModelClass(in_c, hidden=64, out_c=out_c, heads=4)
            else:
                model = ModelClass(in_c, hidden=64, out_c=out_c)
            model = train(model, data, epochs=200, verbose=True)
            metrics = test(model, data)
            results_table.append({
                'Dataset': name,
                'Model': ModelClass.__name__,
                'Train Acc': metrics['train']['acc'],
                'Train Prec': metrics['train']['prec'],
                'Train Rec': metrics['train']['rec'],
                'Train F1': metrics['train']['f1'],
                'Val Acc': metrics['val']['acc'],
                'Val Prec': metrics['val']['prec'],
                'Val Rec': metrics['val']['rec'],
                'Val F1': metrics['val']['f1'],
                'Test Acc': metrics['test']['acc'],
                'Test Prec': metrics['test']['prec'],
                'Test Rec': metrics['test']['rec'],
                'Test F1': metrics['test']['f1'],
            })

    except Exception as e:
        print(f"[ERRO ao processar {name}]: {e}")

# Printar tabela de resultados
print("\n==== Resultados ====")
df_results = pd.DataFrame(results_table)
print(df_results.to_string(index=False, float_format="%.4f"))