import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")