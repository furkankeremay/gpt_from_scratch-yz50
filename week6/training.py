import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import random
from layers import *

words = open('week6/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {'.':0} | {s:i for i,s in enumerate(chars, start = 1)}
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)

random.seed(42)
random.shuffle(words)

block_size = 8

def build_dataset(words):
    X, Y = [], []
    for names in words:
        context = [0] * block_size
        for ch in names + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

torch.manual_seed(42)

n_embd = 24
n_hidden = 128

model = Sequential([ Embedding(vocab_size, n_embd),
                    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
                    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
                    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
                    Linear(n_hidden, vocab_size)])

with torch.no_grad():
    model.layers[-1].weight *= 0.1 

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]   

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.show()

torch.save({'layers': model.layers}, 'week6/model_weights.pth')