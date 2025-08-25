import torch

print("CUDA disponível? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Memória total da GPU (MB):", torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    print("Tensor exemplo na GPU:", torch.randn(2, 3).to("cuda"))
    