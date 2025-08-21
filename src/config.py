# src/config.py

# --- Nova Configuração ---
OUTPUT_DIR = "output" # Nome da pasta principal para salvar os experimentos

# --- Configurações do Modelo ---
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
GAT_HEADS = 4

# --- Configurações de Treinamento ---
EPOCHS = 500
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4

# --- Configurações dos Dados ---
DATASETS_INFO = {
    "Musae-Facebook": {
        "edges_path": "data/musae-facebook/facebook_large/musae_facebook_edges.csv",
        "target_path": "data/musae-facebook/facebook_large/musae_facebook_target.csv",
        "target_col": 'page_type'
    },
    "Musae-Github": {
        "edges_path": "data/musae-github/git_web_ml/musae_git_edges.csv",
        "target_path": "data/musae-github/git_web_ml/musae_git_target.csv",
        "target_col": 'ml_target'
    },
}

# --- Configurações da Execução ---
DATASET_TO_RUN = ["Musae-Github"]