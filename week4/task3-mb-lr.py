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


for isim in names[:5]:
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

for p in parameters:
        p.requires_grad = True


lre = torch.linspace(-3, 0, 1001)
lrs = 10**lre


print('-----------')
print("minibatch training'i")
print('-----------')

lri = []
lossi = []

for i in range(1001):

    ix = torch.randint(0, X.shape[0], (32,))

    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32,100)
    logits = h @ W2 + b2 # (32,27)
    loss = F.cross_entropy(logits, Y[ix])
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = lrs[i]
    for p in parameters:
        p.data += -lr * p.grad

    lri.append(lre[i])
    lossi.append(loss.item())

    if i % 100 == 0:
        print(f'{i}. adım loss: {loss.item():.4f}')

print('-----------')

print(f'son loss: {loss.item():.4f}')

print('-----------')

plt.plot(lri,lossi)
plt.show()



