import torch
import torch.nn.functional as F 
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

checkpoint = torch.load('week6/model_weights.pth', weights_only=False)
layers = checkpoint['layers']
model = Sequential(layers)

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
    print(f'{split} loss: {loss.item():.4f}')

split_loss('train')
split_loss('val')

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
    print (''.join(itos[i] for i in out))