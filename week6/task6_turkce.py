import torch
import torch.nn.functional as F
import random
from layers import *


words = open('week6/isimler.txt', 'r', encoding='utf-8').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {'.':0} | {s:i for i,s in enumerate(chars, start=1)}
itos = {i:s for s, i in stoi.items()}
vocab_size = len(stoi)

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
    return torch.tensor(X), torch.tensor(Y)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


torch.manual_seed(42)
n_embd = 24
n_hidden = 128

model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1 

parameters = model.parameters()
for p in parameters:
    p.requires_grad = True


max_steps = 200000
batch_size = 32

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

    if i % 20000 == 0:
        print(f"  Adım {i:6d}/{max_steps}: Loss = {loss.item():.4f}")

for layer in model.layers:
    layer.training = False

@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    return loss.item()

train_loss = split_loss('train')
dev_loss = split_loss('val')

print(f"Eğitim Tamamlandı!")
print(f"Train Loss: {train_loss:.4f}")
print(f"Dev Loss:   {dev_loss:.4f}")

print("\n✨ Üretilen Türkçe İsimler:")
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))