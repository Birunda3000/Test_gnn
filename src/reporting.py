# reporting.py

"""
Módulo para gerar gráficos e relatórios de texto dos resultados do treinamento.
"""
import os
import matplotlib.pyplot as plt
from src import config

def plot_history(history, model_name, save_dir):
    """Salva um gráfico da acurácia vs. épocas."""
    plt.figure(figsize=(10, 6))
    epochs_ran = range(1, len(history['train_acc']) + 1)
    
    plt.plot(epochs_ran, history['train_acc'], label='Acurácia de Treino')
    plt.plot(epochs_ran, history['val_acc'], label='Acurácia de Validação')
    
    plt.title(f'Histórico de Treinamento - {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f'{model_name}_accuracy_plot.png'
    plt.savefig(os.path.join(save_dir, plot_filename))
    plt.close()
    return plot_filename

def generate_report(results, run_dir):
    """Gera o gráfico e o arquivo de texto com o resumo dos resultados."""
    report_path = os.path.join(run_dir, 'report.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"{'='*30}\n RELATÓRIO GERAL DA EXECUÇÃO\n{'='*30}\n")
        f.write(f"Início da Execução: {results['start_time_str']}\n")
        
        for model_name, data in results['models'].items():
            f.write(f"\n\n--- Modelo: {model_name} ---\n")
            f.write(f"Tempo de Treino: {data['training_time']:.2f} segundos\n")
            f.write(f"Melhor Época (Validação): {data['best_epoch']}\n")
            
            # Gerar e salvar o gráfico de acurácia
            plot_filename = plot_history(data['history'], model_name, run_dir)
            f.write(f"Gráfico de Acurácia: {plot_filename}\n")
            
            f.write("\n--- Métricas Finais (com o melhor modelo de validação) ---\n")
            for split in ["Treino", "Validação", "Teste"]:
                metrics = data['final_metrics'][split]
                f.write(f"  {split}:\n")
                f.write(f"    Acurácia: {metrics['accuracy']:.4f}\n")
                f.write(f"    F1-Score: {metrics['f1']:.4f}\n")
                f.write(f"    Recall:   {metrics['recall']:.4f}\n")
        
        f.write(f"\n\n{'='*30}\n HIPERPARÂMETROS GERAIS\n{'='*30}\n")
        f.write(f"Épocas: {config.EPOCHS}\n")
        f.write(f"Taxa de Aprendizado: {config.LEARNING_RATE}\n")
        f.write(f"Decaimento de Peso: {config.WEIGHT_DECAY}\n")
        f.write(f"Canais Ocultos: {config.HIDDEN_CHANNELS}\n")

    print(f"Relatório salvo em: {report_path}")