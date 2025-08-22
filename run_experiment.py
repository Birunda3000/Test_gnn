# run_experiment.py

import os
import torch
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Importando nossos módulos
from src import config
from src import data_loader
from src import models
from src import engine


def generate_report(exp_dir, results_df):
    """Gera o arquivo relatorio.txt com as configurações e resultados."""
    report_path = os.path.join(exp_dir, "relatorio.txt")
    now = datetime.now().astimezone()
    tz_str = now.strftime("%Z%z")
    with open(report_path, "w") as f:
        f.write("=" * 40 + "\n")
        f.write("### RELATÓRIO DO EXPERIMENTO ###\n")
        f.write("=" * 40 + "\n\n")

        f.write("CONFIGURAÇÕES:\n")
        f.write(f"  - Timestamp: {now.strftime('%d/%m/%Y %H:%M:%S')} ({tz_str})\n")
        f.write(f"  - Embedding Dim: {config.EMBEDDING_DIM}\n")
        f.write(f"  - Hidden Dim: {config.HIDDEN_DIM}\n")
        f.write(f"  - Num Layers: {config.NUM_LAYERS}\n")
        f.write(f"  - Dropout: {config.DROPOUT}\n")
        f.write(f"  - Epochs: {config.EPOCHS}\n")
        f.write(f"  - Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"  - Weight Decay: {config.WEIGHT_DECAY}\n")
        f.write(f"  - Patience (Early Stop): {config.PATIENCE}\n\n")

        f.write("RESULTADOS POR MODELO:\n")
        f.write(results_df.to_string(index=False, float_format="%.4f"))
        f.write("\n")


def generate_accuracy_plot(history, model_name, output_dir):
    """Gera e salva o gráfico de acurácia vs. épocas."""
    plt.figure(figsize=(12, 7))

    # Verifica se o histórico não está vazio para evitar erros
    if not history["train_acc"]:
        print(f"Histórico vazio para {model_name}, pulando a geração do gráfico.")
        return

    epochs_recorded = len(history["train_acc"])
    x_axis = range(0, epochs_recorded * 10, 10)

    plt.plot(
        x_axis, history["train_acc"], label="Acurácia de Treino", marker="o", alpha=0.7
    )
    plt.plot(
        x_axis,
        history["val_acc"],
        label="Acurácia de Validação",
        marker="s",
        linewidth=1.5,
        alpha=0.7
    )
    plt.plot(
        x_axis,
        history["test_acc"],
        label="Acurácia de Teste (Monitorada)",
        marker="x",
        linestyle="--",
        linewidth=2.5,
    )

    # ==========================================================
    # MUDANÇA: A linha vertical agora marca a melhor acurácia de TESTE
    # ==========================================================
    best_test_epoch_index = np.argmax(history["test_acc"])
    best_test_acc = history["test_acc"][best_test_epoch_index]
    best_epoch = x_axis[best_test_epoch_index]
    plt.axvline(
        x=best_epoch,
        color="r",
        linestyle="--",
        label=f"Melhor Época ({best_epoch}) - Test Acc: {best_test_acc:.2f}",
    )

    plt.title(f"Acurácia vs. Épocas - {model_name}")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{model_name}_accuracy_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico de acurácia para {model_name} salvo em: {plot_path}")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results_list = []

    for name in config.DATASET_TO_RUN:
        try:
            info = config.DATASETS_INFO[name]
            data = data_loader.load_data(
                edges_path=info["edges_path"],
                target_path=info["target_path"],
                target_col=info["target_col"],
            )

            print(f"\n=== Dataset: {name} ===")
            print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
            print(
                f"Features: {config.EMBEDDING_DIM} (aprendidas), Classes: {int(data.y.max().item()) + 1}"
            )

            out_c = int(data.y.max().item()) + 1

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            tmp_exp_folder_name = f"_tmp_{timestamp}"
            tmp_exp_dir = os.path.join(config.OUTPUT_DIR, tmp_exp_folder_name)
            os.makedirs(tmp_exp_dir, exist_ok=True)
            models_dir = os.path.join(tmp_exp_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            for model_name, ModelClass in models.ALL_MODELS.items():
                print(f"\n--- Treinando {model_name} ---")

                model_params = {
                    "num_nodes": data.num_nodes,
                    "embedding_dim": config.EMBEDDING_DIM,
                    "hidden": config.HIDDEN_DIM,
                    "out_c": out_c,
                }
                
                if model_name in ["GCNNet", "SAGENet", "GINNet", "GATNet"]:
                    model_params["num_layers"] = config.NUM_LAYERS
                if model_name == "GATNet":
                    model_params["heads"] = config.GAT_HEADS
                if model_name == "APPNPNet":
                    model_params["K"] = config.APPNP_K
                    model_params["alpha"] = config.APPNP_ALPHA
                if model_name == "SGConvNet":
                    model_params["K"] = config.SGCONV_K
                    del model_params["hidden"]
                
                model = ModelClass(**model_params)

                model, train_time, history, best_metrics = engine.train(
                    model=model,
                    data=data,
                    epochs=config.EPOCHS,
                    learning_rate=config.LEARNING_RATE,
                    weight_decay=config.WEIGHT_DECAY,
                    patience=config.PATIENCE,
                    verbose=True,
                )

                # A função 'test' ainda é útil para obter F1, Recall, etc.
                metrics = engine.test(model, data)

                # Usamos a acurácia de teste salva em 'best_metrics' para o nome do arquivo
                test_acc = best_metrics["test_acc"]
                model_filename = f"{model_name}__test_acc_{test_acc:.4f}.pt"
                model_path = os.path.join(models_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                print(f"Modelo salvo em: {model_path}")

                generate_accuracy_plot(history, model_name, tmp_exp_dir)

                # Para o relatório final, também usamos os dados do melhor momento
                results_list.append(
                    {
                        "Model": model_name,
                        "Train Time (s)": train_time,
                        "Train Acc": best_metrics["train_acc"],
                        "Train F1": metrics["train"]["f1"], # F1, etc, são do modelo final
                        "Train Recall": metrics["train"]["recall"],
                        "Val Acc": best_metrics["val_acc"],
                        "Val F1": metrics["val"]["f1"],
                        "Val Recall": metrics["val"]["recall"],
                        "Test Acc": best_metrics["test_acc"],
                        "Test F1": metrics["test"]["f1"],
                        "Test Recall": metrics["test"]["recall"],
                    }
                )
        except Exception as e:
            print(f"[ERRO ao processar {name}]: {e}")

    if not results_list:
        print("Nenhum resultado foi gerado. Encerrando.")
        return

    results_df = pd.DataFrame(results_list)
    best_test_acc = results_df["Test Acc"].max()
    best_test_acc_str = f"{best_test_acc:.4f}"

    final_exp_folder_name = f"{timestamp}__test_acc_{best_test_acc_str}"
    final_exp_dir = os.path.join(config.OUTPUT_DIR, final_exp_folder_name)
    os.rename(tmp_exp_dir, final_exp_dir)

    generate_report(final_exp_dir, results_df)
    print(f"Relatório salvo em: {os.path.join(final_exp_dir, 'relatorio.txt')}")

    print("\n\n==== Resultados Finais ====")
    print(results_df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()