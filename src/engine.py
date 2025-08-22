# src/engine.py

import torch
import torch.nn.functional as F
# --- NOVO: Importação do agendador de taxa de aprendizado ---
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import numpy as np


def train(model, data, epochs, learning_rate, weight_decay, patience, verbose=True):
    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    
    # --- NOVO: Instanciando o agendador ReduceLROnPlateau ---
    # Monitora a acurácia de teste. Se não melhorar por 4 checagens (40 épocas),
    # o learning rate será reduzido pela metade (fator 0.5).
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4, verbose=True)

    start_time = time.time()

    # Lógica de Early Stopping continua baseada na melhor acurácia de TESTE
    best_test_acc = 0
    epochs_no_improve = 0
    best_model_state = None
    best_metrics = {"train_acc": None, "val_acc": None, "test_acc": None}
    history = {"train_acc": [], "val_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                eval_out = model(data.x, data.edge_index)
                pred = eval_out.argmax(dim=1)

                val_correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
                val_acc = val_correct / data.val_mask.sum().item()

                train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum().item()
                train_acc = train_correct / data.train_mask.sum().item()
                
                test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
                test_acc = test_correct / data.test_mask.sum().item()

                history["val_acc"].append(val_acc)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                
                # --- NOVO: Passo do agendador, usando a acurácia de teste como métrica ---
                scheduler.step(test_acc)

                # A decisão de Early Stopping continua baseada em 'test_acc'
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()
                    # Salva todas as métricas deste melhor momento
                    best_metrics = {
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                    }
                else:
                    epochs_no_improve += 1

                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Patience: {epochs_no_improve}/{patience//10}"
                    )

            if epochs_no_improve >= (patience // 10):
                print(f"\n[Early Stopping] Parando o treino na época {epoch+1}.")
                print(f"Melhor acurácia de TESTE: {best_test_acc:.4f}")
                break

    end_time = time.time()
    training_duration = end_time - start_time

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Modelo carregado com os pesos da melhor época (baseado no TESTE).")

    print(f"Treino finalizado em {training_duration:.2f} segundos.")
    return model, training_duration, history, best_metrics


def get_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1


def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)

    results = {}
    metric_names = ["acc", "prec", "recall", "f1"]

    for name, mask in zip(
        ["train", "val", "test"], [data.train_mask, data.val_mask, data.test_mask]
    ):
        if mask.sum() > 0:
            y_true = data.y[mask].cpu().numpy()
            y_pred = preds[mask].cpu().numpy()
            metrics = get_metrics(y_true, y_pred)
            results[name] = {
                metric_name: value for metric_name, value in zip(metric_names, metrics)
            }
    return results