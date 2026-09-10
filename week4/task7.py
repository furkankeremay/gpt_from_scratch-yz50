import torch
import torch.nn.functional as F
import random

isimler = open('week3/isimler.txt', encoding='utf-8').read().splitlines()
isimler = [isim.replace(' ', '-') for isim in isimler]

chars = sorted(list(set(''.join(isimler))))
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
random.shuffle(isimler)
n1 = int(len(isimler) * 0.8)
n2 = int(len(isimler) * 0.9)

Xtr, Ytr = build_dataset(isimler[:n1])
Xdev, Ydev = build_dataset(isimler[n1:n2])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, 10), generator=g)
W1 = torch.randn((30, 300), generator=g) * ((5.0/3.0) / (30**0.5)) # Kaiming Init
b1 = torch.randn(300, generator=g) * 0.01
W2 = torch.randn((300, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0
parameters = [C, W1, b1, W2, b2]

for p in parameters: 
    p.requires_grad = True

print("Türkçe MLP Modeli Eğitiliyor (Bu işlem birkaç saniye sürebilir)...")
for i in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    
    for p in parameters: p.grad = None
    loss.backward()
    lr = 0.1 if i < 15000 else 0.01
    for p in parameters: p.data += -lr * p.grad

print(f'MLP Türkçe Train Loss : {loss:.4f}')
emb_dev = C[Xdev]
logits_dev = torch.tanh(emb_dev.view(-1, 30) @ W1 + b1) @ W2 + b2
loss_dev = F.cross_entropy(logits_dev, Ydev)
print(f"\nMLP Türkçe Dev Loss: {loss_dev.item():.4f}\n")

W_bigram = torch.load('week3/bigram_agirliklari_tr.pt')

print(f"{'ESKİ BİGRAM MODELİ (Türkçe)':<35} | {'YENİ MLP MODELİ (Türkçe)'}")
print("-" * 65)

g_gen = torch.Generator().manual_seed(2147483647)

for _ in range(15):
    out_bi = []
    ix_bi = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix_bi]), num_classes=vocab_size).float()
        logits_bi = xenc @ W_bigram
        probs_bi = F.softmax(logits_bi, dim=1)
        ix_bi = torch.multinomial(probs_bi, num_samples=1, replacement=True, generator=g_gen).item()
        out_bi.append(itos[ix_bi])
        if ix_bi == 0: break
    isim_bi = ''.join(out_bi)
    
    out_mlp = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits_mlp = h @ W2 + b2
        probs_mlp = F.softmax(logits_mlp, dim=1)
        ix_mlp = torch.multinomial(probs_mlp, num_samples=1, generator=g_gen).item()
        context = context[1:] + [ix_mlp]
        out_mlp.append(itos[ix_mlp])
        if ix_mlp == 0: break
    isim_mlp = ''.join(out_mlp)
    
    print(f"{isim_bi:<35} | {isim_mlp}")