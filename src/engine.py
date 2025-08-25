# engine.py

"""
Contém as funções para o loop de treinamento e avaliação do modelo.
"""

import torch

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
    """Avalia o modelo nos conjuntos de treino, validação e teste."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = (out > 0.5).float()
        
        accs = {}
        for name, mask in [("Treino", data.train_mask), ("Validação", data.val_mask), ("Teste", data.test_mask)]:
            correct = (pred[mask] == data.y[mask]).sum()
            total = mask.sum()
            accs[name] = int(correct) / int(total)
    return accs