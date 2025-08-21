# Teste de GNNs em Grafos Reais

Este repositório é um **fork** do projeto original [IvoAP/Test_gnn](https://github.com/IvoAP/Test_gnn) e tem como objetivo facilitar experimentos e comparações entre diferentes arquiteturas de Graph Neural Networks (GNNs) em datasets reais de grafos, como Musae-Facebook e Musae-Github.

## Principais Funcionalidades

- **Treinamento e avaliação** de múltiplos modelos GNN (GCN, GraphSAGE, GAT, GIN, APPNP, SGConv) em datasets reais.
- **Configuração centralizada** de hiperparâmetros em `src/config.py` (embedding, hidden, dropout, epochs, learning rate, etc).
- **Relatórios automáticos** com métricas de treino, validação e teste, incluindo timestamp e fuso horário.
- **Salvamento organizado** dos modelos e resultados em pastas nomeadas automaticamente.
- **Pipeline pronto para rodar em CPU ou GPU** (via Docker e Docker Compose).

## Estrutura do Projeto

```
workspace/
├── data/                # Datasets Musae-Facebook e Musae-Github
├── output/              # Resultados e modelos treinados
├── src/
│   ├── config.py        # Configurações e hiperparâmetros
│   ├── data_loader.py   # Funções de carregamento dos dados
│   ├── models.py        # Definição das arquiteturas GNN
│   ├── engine.py        # Funções de treino e avaliação
├── run_experiment.py    # Script principal de experimentos
├── main.py              # Script alternativo e testes
├── main2.py             # Script alternativo com embeddings aprendidas
├── requirements.txt     # Dependências Python
├── Dockerfile           # Ambiente Docker pronto para GPU
├── docker-compose.yml   # Orquestração do ambiente
```

## Como Usar

### 1. Clonar o Repositório

```bash
git clone https://github.com/SEU_USUARIO/Test_gnn.git
cd Test_gnn
```

### 2. Instalar Dependências

Recomenda-se o uso do Docker para garantir o ambiente correto (inclusive suporte a GPU).

**Com Docker:**

```bash
docker-compose up --build
```

**Ou localmente (fora do Docker):**

```bash
pip install -r requirements.txt
```

### 3. Configurar Experimentos

Edite o arquivo `src/config.py` para ajustar hiperparâmetros, datasets e demais opções.

### 4. Executar Experimentos

Dentro do container ou ambiente Python:

```bash
python run_experiment.py
```

Os resultados e modelos serão salvos em `output/`.

### 5. Ver Relatórios

Após a execução, consulte o arquivo `relatorio.txt` dentro da pasta `output/<timestamp>__test_acc_xxxx/` para ver métricas detalhadas.

## Datasets

Os experimentos utilizam os datasets públicos do projeto Musae:

- [Musae-Facebook](https://snap.stanford.edu/data/facebook-large-page-page-network.html)
- [Musae-Github](https://snap.stanford.edu/data/github-social.html)

Os arquivos CSV devem estar nas pastas conforme configurado em `src/config.py`.

## Créditos

- Projeto original: [IvoAP/Test_gnn](https://github.com/IvoAP/Test_gnn)
- Este fork: melhorias de automação, organização, relatórios e suporte Docker.

---

