# config.py

"""
Arquivo central de configurações para o projeto GNN.
"""
import os
import torch

# --- 1. Configurações de Caminhos (Paths) ---
# Diretório base onde os dados estão localizados
DATA_DIR = "data/musae-github/git_web_ml"
# Nomes dos arquivos de dados
FEATURES_FILE = os.path.join(DATA_DIR, "musae_git_features.json")
EDGES_FILE = os.path.join(DATA_DIR, "musae_git_edges.csv")
TARGET_FILE = os.path.join(DATA_DIR, "musae_git_target.csv")

# --- 2. Configurações de Treinamento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 2001
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4

# --- 3. Configurações do Modelo ---
# Parâmetros que são compartilhados entre a maioria dos modelos
HIDDEN_CHANNELS = 16
# Parâmetros específicos para alguns modelos
GAT_HEADS = 4

# --- 4. Configurações da Divisão dos Dados ---
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
# A proporção de teste será o que sobrar (1.0 - 0.8 - 0.1 = 0.1)

RESULTS_DIR = "runs"