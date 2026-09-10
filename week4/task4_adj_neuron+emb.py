import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json


kayit_yolu1 = "week4/loss_sonuclari1.json"

with open(kayit_yolu1, "r", encoding="utf-8") as f:
    loss_veriler1 = json.load(f)

kayit_yolu2 = "week4/loss_sonuclari2.json"

with open(kayit_yolu2, "r", encoding="utf-8") as f:
    loss_veriler2 = json.load(f)



print('-----------')
print("100 Neuron + 2 boyutlu embedding'li loss değerleri")
print(f"Train Loss = {loss_veriler1['train_loss']:.4f}")
print(f"Dev Loss = {loss_veriler1['dev_loss']:.4f}")
print('-----------')



print("300 Neuron + 2 boyutlu embedding'li loss değerleri")
print(f"Train Loss = {loss_veriler2['train_loss']:.4f}")
print(f"Dev Loss = {loss_veriler2['dev_loss']:.4f}")
print('-----------')


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
C = torch.randn((27,10), generator=g)
W1 = torch.randn((30,300), generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn((300,27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C,W1,b1,W2,b2]
total_parameters = sum(p.nelement() for p in parameters)

for p in parameters:
        p.requires_grad = True




lossi = []
stepi = []

print('Training  Başladı')

for i in range(200000):

    ix = torch.randint(0, Xtr.shape[0], (32,))

    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32,100)
    logits = h @ W2 + b2 # (32,27)
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i<100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.item())
    stepi.append(i)

emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32,100)
logits = h @ W2 + b2 # (32,27)
losstr = F.cross_entropy(logits, Ytr)


print("300 Neuron + 10 embedding'li loss değerleri")

print(f'Training set loss: {losstr.item():.4f}')


emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32,100)
logits = h @ W2 + b2 # (32,27)
lossdev = F.cross_entropy(logits, Ydev)

print(f'Validation set loss: {lossdev.item():.4f}')

print('-----------')

loss_verileri = {
    "train_loss": losstr.item(), 
    "dev_loss": lossdev.item(),   
}

kayit_yolu = "week4/loss_sonuclari3.json"
with open(kayit_yolu, "w", encoding="utf-8") as f:
    json.dump(loss_verileri, f, indent=4)



agirliklar = {
    'C': C,
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2
}
torch.save(agirliklar, 'week4/mlp_agirliklari.pt')
print("Model ağırlıkları 'week4/mlp_agirliklari.pt' olarak başarıyla kaydedildi.")


plt.plot(stepi, lossi)
plt.show()

