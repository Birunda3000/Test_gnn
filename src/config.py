# src/config.py

# --- Configurações do Modelo ---
EMBEDDING_DIM = 256  # Dimensão das features que serão aprendidas
HIDDEN_DIM = 128     # Dimensão das camadas ocultas
GAT_HEADS = 4        # Número de heads para o modelo GATNet

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
# Lista de datasets para rodar. Ex: ["Musae-Github", "Musae-Facebook"]
DATASET_TO_RUN = ["Musae-Github"]