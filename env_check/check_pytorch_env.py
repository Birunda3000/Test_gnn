# check_pytorch_env.py
# -*- coding: utf-8 -*-
"""
Script de Diagnóstico do Ambiente PyTorch (GPU, Docker, Performance)
Baseado no script de diagnóstico para TensorFlow.
"""
import os
import sys
import platform
import time
import numpy as np
import psutil

# Importações específicas do PyTorch
import torch
import torch.backends.cudnn as cudnn
from torch_geometric import __version__ as pyg_version

# --- Funções Auxiliares (mantidas e adaptadas) ---
def formatar_bytes(b):
    if b < 1024: return f"{b} Bytes"
    elif b < 1024**2: return f"{b/1024:.2f} KB"
    elif b < 1024**3: return f"{b/1024**2:.2f} MB"
    elif b < 1024**4: return f"{b/1024**3:.2f} GB"
    else: return f"{b/1024**4:.2f} TB"

def imprimir_cabecalho(titulo):
    print("\n" + "=" * 80)
    print(f"### {titulo.upper()} ###")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

def verificar_execucao_docker():
    is_docker = os.path.exists('/.dockerenv')
    print(f"\n[INFO] Ambiente de Execução: {'Container Docker' if is_docker else 'Sistema Host (Não-Docker)'}")
    return is_docker

def verificar_recursos_hardware():
    print("\n[INFO] Verificando recursos de hardware...")
    print(f"  - CPU Modelo: {platform.processor()}")
    print(f"  - Núcleos Físicos: {psutil.cpu_count(logical=False)}")
    print(f"  - Núcleos Lógicos: {psutil.cpu_count(logical=True)}")
    mem = psutil.virtual_memory()
    print(f"  - Memória RAM Total: {formatar_bytes(mem.total)} (Usada: {mem.percent}%)")

def verificar_versoes_sistema():
    print("\n[INFO] Verificando versões do sistema e bibliotecas...")
    print(f"  - Versão do Python: {sys.version.split()[0]}")
    print(f"  - Sistema Operacional: {platform.system()} {platform.release()}")
    # [PYTORCH] Verificações específicas do PyTorch e PyG
    print(f"  - Versão do PyTorch: {torch.__version__}")
    print(f"  - Versão do PyTorch Geometric: {pyg_version}")

def verificar_dispositivos_gpu_pytorch():
    print("\n[INFO] Procurando por dispositivos GPU (CUDA)...")
    gpus_disponiveis = torch.cuda.is_available()
    if gpus_disponiveis:
        gpu_count = torch.cuda.device_count()
        print(f"  - [SUCESSO] {gpu_count} GPU(s) com CUDA encontrada(s):")
        for i in range(gpu_count):
            print(f"    - GPU[{i}]: {torch.cuda.get_device_name(i)}")
        # [PYTORCH] Informações de build do CUDA/cuDNN
        print(f"    - Versão CUDA (Runtime): {torch.version.cuda}")
        print(f"    - Versão cuDNN (Runtime): {cudnn.version()}")
    else:
        print("  - [AVISO] Nenhuma GPU com CUDA foi encontrada pelo PyTorch.")
    return gpus_disponiveis

def verificar_otimizacoes_avancadas_pytorch():
    print("\n[INFO] Verificando otimizações avançadas do PyTorch...")
    # 1. Verificação do compilador (torch.compile) - Equivalente moderno do JIT/XLA
    if hasattr(torch, 'compile'):
        try:
            @torch.compile
            def fn_teste_compile(x, y):
                return x + y
            fn_teste_compile(torch.randn(2), torch.randn(2))
            print("  - [SUCESSO] Compilador (torch.compile) está disponível e funcional.")
        except Exception as e:
            print(f"  - [AVISO] torch.compile está disponível mas falhou no teste: {e}")
    else:
        print("  - [INFO] Compilador (torch.compile) não disponível (requer PyTorch 2.0+).")

    # 2. Verificação de Mixed Precision (AMP)
    if torch.cuda.is_available():
        try:
            # Verifica suporte a bfloat16, comum em GPUs mais novas (Ampere+)
            suporta_bf16 = torch.cuda.is_bf16_supported()
            print(f"  - [INFO] Suporte a Mixed Precision (bfloat16): {'Sim' if suporta_bf16 else 'Não'}")
            # Testa o autocast com float16
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                a = torch.randn(2, device='cuda')
                b = torch.randn(2, device='cuda')
                _ = a + b
            print("  - [SUCESSO] Mixed Precision (AMP com float16) foi ativada com sucesso.")
        except Exception as e:
            print(f"  - [AVISO] Não foi possível testar Mixed Precision: {e}")

def executar_teste_performance_pytorch(device_str, tamanho_matriz):
    print(f"\n[INFO] Iniciando teste de performance em: {device_str} (Matriz {tamanho_matriz}x{tamanho_matriz})")
    try:
        device = torch.device(device_str)
        matriz_a = torch.randn(tamanho_matriz, tamanho_matriz, device=device)
        matriz_b = torch.randn(tamanho_matriz, tamanho_matriz, device=device)
        
        # Warm-up run
        _ = torch.matmul(matriz_a, matriz_b)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        inicio = time.perf_counter()
        resultado = torch.matmul(matriz_a, matriz_b)
        
        # [PYTORCH] Sincronização é crucial para benchmarks de GPU corretos
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        fim = time.perf_counter()
        duracao = fim - inicio
        print(f"  - [RESULTADO] Tempo de execução em {device_str}: {duracao:.4f} segundos.")
        return duracao
    except Exception as e:
        print(f"  - [ERRO] Falha ao executar o teste em {device_str}: {e}")
        return float('inf')

def imprimir_sumario_final(is_docker, tem_gpu, tempo_gpu, tempo_cpu):
    imprimir_cabecalho("Diagnóstico Final do Ambiente PyTorch")
    print(f"✔️ Ambiente de Execução: {'Container Docker' if is_docker else 'Sistema Host'}.")
    
    if tem_gpu and tempo_gpu < float('inf'):
        print("✔️ Aceleração por GPU: Ativa e Testada.")
        print(f"   - Tempo de execução na GPU: {tempo_gpu:.4f}s")
        print(f"   - Tempo de execução na CPU: {tempo_cpu:.4f}s")
        if tempo_gpu < tempo_cpu:
            speedup = tempo_cpu / tempo_gpu
            print(f"   - Otimização: A GPU é aproximadamente {speedup:.2f}x mais rápida que a CPU.")
            print("\n   >>> [AVALIAÇÃO GERAL]: SUCESSO! Ambiente otimizado para Deep Learning. <<<")
        else:
            print("\n   >>> [AVALIAÇÃO GERAL]: ATENÇÃO! A GPU está ativa, mas foi mais lenta que a CPU no teste. (Verifique a carga do sistema ou o tamanho do teste)")
    else:
        print("❌ Aceleração por GPU: INATIVA ou com falhas.")
        print("\n   >>> [AVALIAÇÃO GERAL]: ATENÇÃO! O ambiente está a usar apenas a CPU. <<<")
    print("=" * 80)

def main():
    imprimir_cabecalho("Início do Diagnóstico do Ambiente PyTorch")
    TAMANHO_DA_MATRIZ_DE_TESTE = 4096 # Um pouco menor para rodar rápido em mais hardwares
    
    is_docker = verificar_execucao_docker()
    verificar_recursos_hardware()
    verificar_versoes_sistema()
    
    gpus_encontradas = verificar_dispositivos_gpu_pytorch()
    if gpus_encontradas:
        verificar_otimizacoes_avancadas_pytorch()
        
    tempo_cpu = executar_teste_performance_pytorch('cpu', tamanho_matriz=TAMANHO_DA_MATRIZ_DE_TESTE)
    tempo_gpu = float('inf')
    if gpus_encontradas:
        tempo_gpu = executar_teste_performance_pytorch('cuda:0', tamanho_matriz=TAMANHO_DA_MATRIZ_DE_TESTE)

    imprimir_sumario_final(is_docker, gpus_encontradas, tempo_gpu, tempo_cpu)

if __name__ == "__main__":
    main()