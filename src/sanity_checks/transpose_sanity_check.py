import torch

# fake tensor shaped like your log_probs: (batch=1, seq_len=3, classes=4)
fake = torch.tensor([[
    [1, 2, 3, 4],   # timestep 0
    [5, 6, 7, 8],   # timestep 1
    [9, 10, 11, 12] # timestep 2
]])

print("original shape:", fake.shape)
print("original:\n", fake)

transposed = fake.transpose(0, 1)
print("transposed shape:", transposed.shape)
print("transposed:\n", transposed)