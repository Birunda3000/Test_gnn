# main.py

"""
Arquivo principal para executar o pipeline de treinamento da GNN.
"""

import torch
from src.data_loader import load_and_prepare_data
from src.model import SimpleGNN
from src.engine import train_one_epoch, evaluate

# --- HIPERPARÂMETROS E CONFIGURAÇÕES ---
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
HIDDEN_CHANNELS = 16

def main():
    # Carregar e preparar os dados
    data = load_and_prepare_data()

    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsando dispositivo: {device}")
    
    data = data.to(device)
    
    # Instanciar modelo, otimizador e função de perda
    model = SimpleGNN(
        num_node_features=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCELoss()

    # Loop de Treinamento
    print("\nIniciando treinamento...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, data, optimizer, criterion)
        
        if epoch % 10 == 0:
            accs = evaluate(model, data)
            print(f'Época: {epoch:03d}, Perda: {loss:.4f}, '
                  f'Acurácia Treino: {accs["Treino"]:.4f}, '
                  f'Acurácia Validação: {accs["Validação"]:.4f}')

    # Avaliação Final
    print("\nTreinamento concluído.")
    final_accs = evaluate(model, data)
    print(f'Acurácia Final de Teste: {final_accs["Teste"]:.4f}')

if __name__ == '__main__':
    main()