# src/config.py

# --- Nova Configuração ---
OUTPUT_DIR = "output" # Nome da pasta principal para salvar os experimentos

# --- Configurações do Modelo ---
EMBEDDING_DIM = 512
HIDDEN_DIM = 256
GAT_HEADS = 8
DROPOUT = 0.5
NUM_LAYERS = 2  # NOVO: Número de camadas para os modelos GNN

# --- Hiperparâmetros Específicos (Opcional, mas recomendado) ---
APPNP_K = 10
APPNP_ALPHA = 0.1
SGCONV_K = 2

# --- Configurações de Treinamento ---
EPOCHS = 20000 # Aumentado para dar espaço ao Early Stopping
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 5e-5
PATIENCE = 1000

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