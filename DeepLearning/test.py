import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Shows bundled CUDA runtime
print(torch.cuda.get_device_name(0))
