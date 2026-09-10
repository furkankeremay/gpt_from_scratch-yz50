import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

names = open('week4/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(names))))

stoi1 = {s:i+1 for i,s in enumerate(chars)}
stoi = {'.': 0, **stoi1}
itos = {i:s for s,i in stoi.items()}

def build_dataset(dataset):
    block_size = 3
    X,Y = [],[]


    for isim in dataset:
        context = [0] * block_size
        for ch in isim + '.':
            ix = stoi[ch]
            #print(''.join(itos[i] for i in context), '--->', ch)
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y 

import random
random.seed(42)
random.shuffle(names)

n1 = int(len(names)*0.8)
n2 = int(len(names)*0.9)

Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xtest , Ytest = build_dataset(names[n2:])



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


print('-----------')
print("minibatch training'i")
print('-----------')

lri = []
lossi = []

for i in range(60000):

    ix = torch.randint(0, Xtr.shape[0], (32,))

    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    if i<30000:
         lr = 0.1
    else:
        lr = 0.01
    for p in parameters:
        p.data += -lr * p.grad

emb = C[Xtr]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32,100)
logits = h @ W2 + b2 # (32,27)
losstr = F.cross_entropy(logits, Ytr)


print('-----------')

print(f'Training set loss: {losstr.item():.4f}')

print('-----------')

emb = C[Xdev]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (32,100)
logits = h @ W2 + b2 # (32,27)
lossdev = F.cross_entropy(logits, Ydev)

print(f'Validation set loss: {lossdev.item():.4f}')

print('-----------')

print('training loss and validation loss is roughly equal so we are underfitting')

print('-----------')

import json

loss_verileri = {
    "train_loss": losstr.item(), 
    "dev_loss": lossdev.item(),  
}

kayit_yolu = "week4/loss_sonuclari1.json"
with open(kayit_yolu, "w", encoding="utf-8") as f:
    json.dump(loss_verileri, f, indent=4)


plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
     plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
plt.grid('minor')
plt.show()




