import pandas as pd
import json

# Carregar o arquivo CSV
try:
    df_target = pd.read_csv('data/musae-github/git_web_ml/musae_git_target.csv')
except FileNotFoundError:
    print("O arquivo 'musae_git_target.csv' não foi encontrado no caminho esperado.")
    df_target = None

# Carregar o arquivo JSON
try:
    with open('data/musae-github/git_web_ml/musae_git_features.json', 'r') as f:
        features_data = json.load(f)
except FileNotFoundError:
    print("O arquivo 'musae_git_features.json' não foi encontrado no caminho esperado.")
    features_data = None

if df_target is not None and features_data is not None:
    # Adicionar as features ao DataFrame
    # A função map vai procurar o 'id' de cada linha do DataFrame como uma chave no dicionário de features
    # É importante converter o 'id' para string para que corresponda às chaves do JSON
    df_target['features'] = df_target['id'].astype(str).map(features_data)

    # Salvar o DataFrame resultante em um novo arquivo CSV
    df_target.to_csv('github_merged.csv', index=False)

    print("Amostra do arquivo 'github_merged.csv' gerado:")
    print(df_target.head())