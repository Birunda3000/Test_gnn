# run_experiment.py

import os
import torch
import pandas as pd
from datetime import datetime

# Importando nossos módulos
from src import config
from src import data_loader
from src import models
from src import engine

def generate_report(exp_dir, results_df):
    """Gera o arquivo relatorio.txt com as configurações e resultados."""
    report_path = os.path.join(exp_dir, "relatorio.txt")
    with open(report_path, "w") as f:
        f.write("="*40 + "\n")
        f.write("### RELATÓRIO DO EXPERIMENTO ###\n")
        f.write("="*40 + "\n\n")
        
        f.write("CONFIGURAÇÕES:\n")
        f.write(f"  - Timestamp: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"  - Embedding Dim: {config.EMBEDDING_DIM}\n")
        f.write(f"  - Hidden Dim: {config.HIDDEN_DIM}\n")
        f.write(f"  - Epochs: {config.EPOCHS}\n")
        f.write(f"  - Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"  - Weight Decay: {config.WEIGHT_DECAY}\n\n")
        
        f.write("RESULTADOS POR MODELO:\n")
        f.write(results_df.to_string(index=False, float_format="%.4f"))
        f.write("\n")

def main():
    """Função principal que orquestra o experimento."""
    # Cria a pasta de output principal, se não existir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Guarda os resultados e modelos temporariamente
    results_list = []
    trained_models = {}

    for name in config.DATASET_TO_RUN:
        try:
            info = config.DATASETS_INFO[name]
            data = data_loader.load_data(
                edges_path=info["edges_path"],
                target_path=info["target_path"],
                target_col=info["target_col"]
            )
            
            print(f"\n=== Dataset: {name} ===")
            print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
            print(f"Features: {config.EMBEDDING_DIM} (aprendidas), Classes: {int(data.y.max().item()) + 1}")

            out_c = int(data.y.max().item()) + 1

            for model_name, ModelClass in models.ALL_MODELS.items():
                print(f"\n--- Treinando {model_name} ---")
                
                model_params = {
                    "num_nodes": data.num_nodes, "embedding_dim": config.EMBEDDING_DIM,
                    "hidden": config.HIDDEN_DIM, "out_c": out_c,
                }
                if model_name == "GATNet":
                    model_params["heads"] = config.GAT_HEADS
                
                model = ModelClass(**model_params)
                
                model, train_time = engine.train(
                    model=model, data=data, epochs=config.EPOCHS,
                    learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                    verbose=True
                )
                
                metrics = engine.test(model, data)
                trained_models[model_name] = model.state_dict()
                
                # Adiciona todas as métricas e o tempo de treino
                results_list.append({
                    'Model': model_name,
                    'Train Time (s)': train_time,
                    'Train Acc': metrics['train']['acc'], 'Train F1': metrics['train']['f1'], 'Train Recall': metrics['train']['recall'],
                    'Val Acc': metrics['val']['acc'], 'Val F1': metrics['val']['f1'], 'Val Recall': metrics['val']['recall'],
                    'Test Acc': metrics['test']['acc'], 'Test F1': metrics['test']['f1'], 'Test Recall': metrics['test']['recall'],
                })
        except Exception as e:
            print(f"[ERRO ao processar {name}]: {e}")

    if not results_list:
        print("Nenhum resultado foi gerado. Encerrando.")
        return

    # --- Lógica para salvar os resultados ---
    results_df = pd.DataFrame(results_list)
    
    # Encontra a melhor acurácia de teste para nomear a pasta
    best_test_acc = results_df['Test Acc'].max()
    
    # Cria o nome da pasta do experimento
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_folder_name = f"{timestamp}__test_acc_{best_test_acc:.4f}"
    exp_dir = os.path.join(config.OUTPUT_DIR, exp_folder_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Salva os modelos
    models_dir = os.path.join(exp_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    for model_name, model_state in trained_models.items():
        model_path = os.path.join(models_dir, f"{model_name}.pt")
        torch.save(model_state, model_path)
    print(f"\nModelos salvos em: {models_dir}")
    
    # Gera o relatório
    generate_report(exp_dir, results_df)
    print(f"Relatório salvo em: {os.path.join(exp_dir, 'relatorio.txt')}")

    print("\n\n==== Resultados Finais ====")
    print(results_df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()