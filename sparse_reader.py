import json
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse

# 1. Defina o caminho para o seu arquivo JSON
features_path = 'data/musae-github/git_web_ml/musae_git_features.json'

# 2. Carregue os dados do arquivo JSON
try:
    with open(features_path, 'r') as f:
        features_data = json.load(f)
    print(f"Arquivo '{features_path}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo '{features_path}' não foi encontrado.")
    print("Por favor, certifique-se de que o arquivo está no mesmo diretório ou forneça o caminho correto.")
    exit()

# 3. Prepare os dados para o MultiLabelBinarizer
# A ferramenta espera uma lista de listas. Vamos garantir uma ordem consistente
# ordenando os nós por seus IDs (chaves do dicionário).
# As chaves no JSON são strings, então convertemos para int para ordenar corretamente.
node_ids = sorted(features_data.keys(), key=int)
# Agora criamos a lista de listas de features na ordem correta.
feature_lists = [features_data[node_id] for node_id in node_ids]

# 4. Use o MultiLabelBinarizer para criar a matriz esparsa
# Esta é a ferramenta perfeita para "multi-hot encoding".
# `sparse_output=True` é o comando CRÍTICO para gerar uma matriz esparsa e economizar memória.
mlb = MultiLabelBinarizer(sparse_output=True)

# 5. Faça o fit (aprender todas as features) e o transform (criar a matriz)
# Este único comando faz todo o trabalho pesado.
X = mlb.fit_transform(feature_lists)

# 6. Inspecione o resultado
print("\n--- Resultados ---")
print(f"Tipo da matriz gerada: {type(X)}")
print(f"Formato (Shape) da matriz (nós, features únicas): {X.shape}")
print(f"Número de nós (linhas): {X.shape[0]}")
print(f"Número total de features únicas (colunas): {X.shape[1]}")

# O atributo `mlb.classes_` armazena o mapeamento: a i-ésima coluna da matriz
# corresponde à feature de ID `mlb.classes_[i]`.
print(f"\nAs primeiras 10 features (colunas) correspondem aos IDs:")
print(mlb.classes_[:10])

# A matriz 'X' agora está pronta para ser usada em um modelo do scikit-learn.
# Por exemplo:
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X, y)  # 'y' seriam os seus labels (web vs ML)