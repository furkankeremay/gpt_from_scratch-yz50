import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

names = open('week4/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(names))))

stoi1 = {s:i+1 for i,s in enumerate(chars)}
stoi = {'.': 0, **stoi1}
itos = {i:s for s,i in stoi.items()}

block_size = 3
X,Y = [],[]


for isim in names:
    context = [0] * block_size
    for ch in isim + '.':
        ix = stoi[ch]
        #print(''.join(itos[i] for i in context), '--->', ch)
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)


g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator=g)
W1 = torch.randn((6,100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100,27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C,W1,b1,W2,b2]
total_parameters = sum(p.nelement() for p in parameters)

emb = C[X].view(-1, 6)

h = torch.tanh(emb @ W1 + b1)
logits = h @ W2 + b2

print('-----------')

counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
loss = -probs[torch.arange(logits.shape[0]), Y].log().mean()

print(f'El ile Hesaplanmış Loss: {loss.item():.4f}')

print('-----------')


loss = F.cross_entropy(logits, Y)
print(f'Cross Entropy ile Hesaplanmış Loss: {loss.item():.4f}')

print('-----------')

for p in parameters:
    p.requires_grad = True

