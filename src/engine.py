# src/engine.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time # Importamos a biblioteca time

def train(model, data, epochs, learning_rate, weight_decay, verbose=True):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    start_time = time.time() # [MUDANÇA] Começa a contar o tempo
    
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        
        if verbose and (epoch % 50 == 0 or epoch == epochs-1):
            with torch.no_grad():
                pred = out[data.train_mask].argmax(dim=1)
                acc = (pred == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

    end_time = time.time() # [MUDANÇA] Termina de contar o tempo
    training_duration = end_time - start_time
    
    print(f"Treino finalizado em {training_duration:.2f} segundos.")
    
    return model, training_duration

def get_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    # Sensibilidade é o mesmo que recall
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
    
    results = {}

    metric_names = ['acc', 'prec', 'recall', 'f1']
    
    for name, mask in zip(['train', 'val', 'test'], [data.train_mask, data.val_mask, data.test_mask]):
        if mask.sum() > 0:
            y_true = data.y[mask].cpu().numpy()
            y_pred = preds[mask].cpu().numpy()
            metrics = get_metrics(y_true, y_pred)
            results[name] = {metric_name: value for metric_name, value in zip(metric_names, metrics)}
    return results