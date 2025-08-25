# engine.py

"""
Contém as funções para o loop de treinamento e avaliação do modelo.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

def train_one_epoch(model, data, optimizer, criterion):
    """Executa uma única época de treinamento."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    """Avalia o modelo e retorna um dicionário de métricas (acc, f1, recall)."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = (out > 0.5).float()
        
        metrics = {}
        for name, mask in [("Treino", data.train_mask), ("Validação", data.val_mask), ("Teste", data.test_mask)]:
            # Move os dados para a CPU para usar com o scikit-learn
            y_true = data.y[mask].cpu()
            y_pred = pred[mask].cpu()
            
            metrics[name] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0)
            }
    return metrics