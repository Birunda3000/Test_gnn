# run_experiment.py

import pandas as pd

# Importando nossos módulos
from src import config
from src import data_loader
from src import models
from src import engine

def main():
    """Função principal que orquestra o experimento."""
    results_table = []

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
                    "num_nodes": data.num_nodes,
                    "embedding_dim": config.EMBEDDING_DIM,
                    "hidden": config.HIDDEN_DIM,
                    "out_c": out_c,
                }
                if model_name == "GATNet":
                    model_params["heads"] = config.GAT_HEADS
                
                model = ModelClass(**model_params)
                
                model = engine.train(
                    model=model,
                    data=data,
                    epochs=config.EPOCHS,
                    learning_rate=config.LEARNING_RATE,
                    weight_decay=config.WEIGHT_DECAY,
                    verbose=True
                )
                
                metrics = engine.test(model, data)
                
                results_table.append({
                    'Dataset': name, 'Model': model_name,
                    'Test Acc': metrics['test']['acc'], 'Test F1': metrics['test']['f1'],
                    'Val Acc': metrics['val']['acc'], 'Val F1': metrics['val']['f1'],
                })
                print(f"Resultados para {model_name}: Test Acc: {metrics['test']['acc']:.4f}, Test F1: {metrics['test']['f1']:.4f}")

        except Exception as e:
            print(f"[ERRO ao processar {name}]: {e}")

    # Printar tabela de resultados
    print("\n\n==== Resultados Finais ====")
    df_results = pd.DataFrame(results_table)
    print(df_results.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()