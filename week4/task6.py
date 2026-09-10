import torch
import torch.nn.functional as F
import random

names = open('week4/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(names))))
stoi = {'.': 0, **{s:i+1 for i,s in enumerate(chars)}}
itos = {i:s for s,i in stoi.items()}
vocab_size = len(stoi)
block_size = 3

def build_dataset(dataset):
    X, Y = [], []
    for isim in dataset:
        context = [0] * block_size
        for ch in isim + '.':
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(names)
n1 = int(len(names) * 0.8)
n2 = int(len(names) * 0.9)

Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xtest, Ytest = build_dataset(names[n2:])

g = torch.Generator().manual_seed(2147483647)
C_no_bn = torch.randn((vocab_size, 10), generator=g)
W1_no_bn = torch.randn((30, 300), generator=g) * ((5.0/3.0) / (30**0.5))
b1_no_bn = torch.randn(300, generator=g) * 0.01
W2_no_bn = torch.randn((300, vocab_size), generator=g) * 0.01
b2_no_bn = torch.randn(vocab_size, generator=g) * 0
params_no_bn = [C_no_bn, W1_no_bn, b1_no_bn, W2_no_bn, b2_no_bn]
for p in params_no_bn: p.requires_grad = True

for i in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C_no_bn[Xtr[ix]]
    hpreact = emb.view(-1, 30) @ W1_no_bn + b1_no_bn
    h = torch.tanh(hpreact)
    logits = h @ W2_no_bn + b2_no_bn
    loss = F.cross_entropy(logits, Ytr[ix])
    
    for p in params_no_bn: p.grad = None
    loss.backward()
    lr = 0.1 if i < 15000 else 0.01
    for p in params_no_bn: p.data += -lr * p.grad

emb_dev = C_no_bn[Xdev]
hpreact_dev = emb_dev.view(-1, 30) @ W1_no_bn + b1_no_bn
logits_dev = torch.tanh(hpreact_dev) @ W2_no_bn + b2_no_bn
loss_dev_no_bn = F.cross_entropy(logits_dev, Ydev)


g = torch.Generator().manual_seed(2147483647)
C_bn = torch.randn((vocab_size, 10), generator=g)
W1_bn = torch.randn((30, 300), generator=g) * ((5.0/3.0) / (30**0.5))

# BatchNorm parametreleri
bngain = torch.ones((1, 300))
bnbias = torch.zeros((1, 300))

bnmean_running = torch.zeros((1, 300))
bnstd_running = torch.ones((1, 300))

W2_bn = torch.randn((300, vocab_size), generator=g) * 0.01
b2_bn = torch.randn(vocab_size, generator=g) * 0

params_bn = [C_bn, W1_bn, bngain, bnbias, W2_bn, b2_bn]
for p in params_bn: p.requires_grad = True

for i in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C_bn[Xtr[ix]]
    
    hpreact = emb.view(-1, 30) @ W1_bn
    
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
        
    h = torch.tanh(hpreact)
    logits = h @ W2_bn + b2_bn
    loss = F.cross_entropy(logits, Ytr[ix])
    
    for p in params_bn: p.grad = None
    loss.backward()
    lr = 0.1 if i < 15000 else 0.01
    for p in params_bn: p.data += -lr * p.grad

emb_dev = C_bn[Xdev]
hpreact_dev = emb_dev.view(-1, 30) @ W1_bn
hpreact_dev = bngain * (hpreact_dev - bnmean_running) / bnstd_running + bnbias
logits_dev_bn = torch.tanh(hpreact_dev) @ W2_bn + b2_bn
loss_dev_bn = F.cross_entropy(logits_dev_bn, Ydev)


# --- 4. SONUÇ KARŞILAŞTIRMASI ---
print('------------------')
print(f"Standart Model Dev Loss  : {loss_dev_no_bn.item():.4f}")
print(f"BatchNorm Model Dev Loss : {loss_dev_bn.item():.4f}")
print('------------------')