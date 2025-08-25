# main.py

import torch
import os
import time
from datetime import datetime
import shutil

from src import config
from src.data_loader import load_and_prepare_data
from src.model import GCNNet, SAGENet, GATNet, GINNet, APPNPNet, SGConvNet
from src.engine import train_one_epoch, evaluate
from src import reporting

def main():
    # --- 1. SETUP INICIAL DA EXECUÇÃO ---
    start_time_obj = datetime.now()
    timestamp = start_time_obj.strftime("%d-%m-%Y_%H-%M-%S")
    tmp_run_dir = os.path.join(config.RESULTS_DIR, f"_tmp_{timestamp}")
    models_dir = os.path.join(tmp_run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Diretório da execução: {tmp_run_dir}")
    
    # Carregar dados
    data = load_and_prepare_data()
    device = torch.device(config.DEVICE)
    data = data.to(device)
    
    models_to_train = {
        'GCN': GCNNet,
        'GraphSAGE': SAGENet,
        'GAT': GATNet,
        'GIN': GINNet,
        'APPNP': APPNPNet,
        'SGConv': SGConvNet
    }
    
    # Dicionário para guardar todos os resultados
    all_results = {
        'start_time_str': start_time_obj.strftime("%d/%m/%Y, %H:%M:%S"),
        'models': {}
    }
    overall_best_val_acc = 0.0

    # --- 2. LOOP DE TREINAMENTO POR MODELO ---
    for model_name, ModelClass in models_to_train.items():
        print(f"\n{'='*20}\n Treinando modelo: {model_name}\n{'='*20}")
        
        # Setup específico do modelo
        model_kwargs = {'in_c': data.num_node_features, 'hidden': config.HIDDEN_CHANNELS, 'out_c': 1}
        if model_name == 'GAT': model_kwargs['heads'] = config.GAT_HEADS
        model = ModelClass(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        criterion = torch.nn.BCELoss()

        # Variáveis de tracking para o modelo atual
        best_val_acc = 0.0
        best_epoch = 0
        best_model_path = ""
        history = {'train_acc': [], 'val_acc': []}
        
        train_start_time = time.time()
        
        # Loop de épocas
        for epoch in range(1, config.EPOCHS):
            loss = train_one_epoch(model, data, optimizer, criterion)
            
            if epoch % 10 == 0:
                metrics = evaluate(model, data)
                val_acc = metrics["Validação"]["accuracy"]
                history['train_acc'].append(metrics["Treino"]["accuracy"])
                history['val_acc'].append(val_acc)
                
                print(f'Época: {epoch:03d}, Perda: {loss:.4f}, Acurácia Validação: {val_acc:.4f}')

                # Salvar o melhor modelo
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    
                    # Remover o modelo antigo salvo, se houver
                    if os.path.exists(best_model_path): os.remove(best_model_path)
                        
                    best_model_path = os.path.join(models_dir, f"{model_name}__val_acc_{best_val_acc:.4f}.pt")
                    torch.save(model.state_dict(), best_model_path)

        training_time = time.time() - train_start_time

        # Carregar o melhor modelo para avaliação final
        best_model_state = torch.load(best_model_path)
        model.load_state_dict(best_model_state)
        final_metrics = evaluate(model, data)
        
        # Guardar todos os resultados
        all_results['models'][model_name] = {
            'training_time': training_time,
            'best_epoch': best_epoch,
            'history': history,
            'final_metrics': final_metrics,
            'best_model_path': best_model_path
        }
        
        if best_val_acc > overall_best_val_acc:
            overall_best_val_acc = best_val_acc

    # --- 3. GERAR RELATÓRIO E FINALIZAR ---
    reporting.generate_report(all_results, tmp_run_dir)
    
    # Renomear a pasta com a melhor acurácia geral
    final_run_dir = os.path.join(config.RESULTS_DIR, f"{timestamp}__best_val_acc_{overall_best_val_acc:.4f}")
    shutil.move(tmp_run_dir, final_run_dir)
    
    print(f"\nExecução finalizada. Resultados salvos em: {final_run_dir}")

if __name__ == '__main__':
    main()