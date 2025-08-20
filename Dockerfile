# Dockerfile

# Imagem base da NVIDIA para PyTorch. Usamos ela pelo CUDA e drivers.
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Instala ferramentas essenciais de compilação.
# Isso é uma boa prática caso o pip precise compilar algo do zero.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho.
WORKDIR /workspace

# Copia o arquivo de dependências Python.
COPY requirements.txt .

# Instala as bibliotecas Python básicas listadas em requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# --- SEÇÃO CORRIGIDA (Abordagem Robusta) ---

# 1. Primeiro, atualizamos o pip para a versão mais recente.
RUN python -m pip install --upgrade pip

# 2. Instalamos uma VERSÃO ESTÁVEL e específica do PyTorch (CPU) e suas dependências.
#    Isso vai sobrescrever a versão alpha que veio na imagem, nos dando um ambiente previsível.
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. AGORA, instalamos o PyTorch Geometric, sabendo exatamente qual versão do PyTorch temos.
#    A URL aponta para os pacotes compilados para o PyTorch 2.3.1 e CPU.
RUN pip install --no-cache-dir torch_geometric \
  -f https://data.pyg.org/whl/torch-2.3.1+cpu.html

# Garante que o ambiente Python está na PATH do usuário.
ENV PATH="/usr/local/bin:/usr/bin:${PATH}"

# Comando que mantém o container rodando.
CMD ["/bin/bash", "-c", "while true; do sleep infinity; done"]