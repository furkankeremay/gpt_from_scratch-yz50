import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 32
head_size = 16

x = torch.randn(B, T, C)

pos_emb = nn.Embedding(T, C)
pos = torch.arange(T)
x = x + pos_emb(pos)

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)

wei = q @ k.transpose(-2, -1)
wei = wei / (head_size ** 0.5)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

print(wei[0])

q_sample = torch.randn(B, T, head_size)
k_sample = torch.randn(B, T, head_size)
wei_no_scale = q_sample @ k_sample.transpose(-2, -1)
wei_with_scale = wei_no_scale / (head_size ** 0.5)

print(F.softmax(torch.tensor([-8.0, 15.0, 3.0, -2.0]), dim=-1))
print(F.softmax(torch.tensor([-8.0, 15.0, 3.0, -2.0]) / (head_size ** 0.5), dim=-1))