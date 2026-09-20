import torch
import torch.nn.functional as F 
import random

# ---------------------------------------------------------------------------
# 1. VERİ SETİ HAZIRLIĞI
# ---------------------------------------------------------------------------
words = open('week5/names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {'.':0} | {s:i for i,s in enumerate(chars, start = 1)}
itos = {i:s for s, i in stoi.items()}

vocab_size = len(itos)
block_size = 3

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

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# ---------------------------------------------------------------------------
# 2. PARAMETRE İLKLENDİRME (n_hidden 200 yapıldı)
# ---------------------------------------------------------------------------
n_embd = 10
n_hidden = 200

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator = g)
W1 = torch.randn((n_embd * block_size, n_hidden) , generator = g) * (5/3)/(n_embd * block_size)**0.5
b1 = torch.randn((n_hidden),                       generator = g) * 0.1
W2 = torch.randn((n_hidden,          vocab_size) , generator = g) * 0.1
b2 = torch.randn((vocab_size),                     generator = g) * 0.1
bngain = torch.randn((1, n_hidden))*0.1+1.0
bnbias = torch.randn((1,n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(f'Toplam Parametre Sayısı: {sum(p.nelement() for p in parameters)}') 

# ---------------------------------------------------------------------------
# 3. MANUEL GRADIENT İLE EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# ---------------------------------------------------------------------------
max_steps = 200000
batch_size = 32

for step in range(max_steps):
    
    # Minibatch oluşturma
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g)
    Xb, Yb = Xtr[ix], Ytr[ix] 

    # İleri Besleme (Forward Pass)
    emb = C[Xb] 
    embcat = emb.view(emb.shape[0], -1) 
    
    hprebn = embcat @ W1 + b1 
    
    bnmeani = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = (hprebn - bnmeani) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    
    h = torch.tanh(hpreact) 
    
    logits = h @ W2 + b2 
    loss = F.cross_entropy(logits, Yb)

    # Geri Yayılım (Manual Backward Pass)
    dlogits = F.softmax(logits, 1)
    dlogits[range(batch_size), Yb] -= 1
    dlogits /= batch_size

    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)

    dhpreact = (1.0 - h**2) * dh

    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    dhprebn = bngain * bnvar_inv/batch_size * (batch_size*dhpreact - dhpreact.sum(0) - batch_size/(batch_size-1)*bnraw*(dhpreact*bnraw).sum(0))

    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)

    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    
    # Hızlı ve Vektörize edilmiş Embedding Geri Yayılımı (index_add_ kullanımı)
    dC.index_add_(0, Xb.view(-1), demb.view(-1, n_embd))
            
    grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

    # Parametre Güncelleme (Gradient Descent)
    lr = 0.1 if step < 100000 else 0.01
    for p, grad in zip(parameters, grads):
        p.data -= lr * grad 

    # Log yazdırma
    if step % 10000 == 0:
        print(f'{step:7d}/{max_steps:7d}: {loss.item():.4f}')

# ---------------------------------------------------------------------------
# 4. BATCHNORM KALİBRASYONU VE TEST (EVALUATION)
# ---------------------------------------------------------------------------
# Eğitim bittikten sonra BatchNorm mean ve std değerlerini tüm veri seti üzerinde kalibre ediyoruz
with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    bnmean_calib = hprebn.mean(0, keepdim=True)
    bnvar_calib = hprebn.var(0, keepdim=True, unbiased=True)

@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    # Kalibre edilmiş istatistikler kullanılıyor
    hpreact = bngain * (embcat @ W1 + b1 - bnmean_calib) * (bnvar_calib + 1e-5)**-0.5 + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f'{split} loss: {loss.item():.4f}')

print("\n--- TEST SONUÇLARI ---")
split_loss('train')
split_loss('val')

# ---------------------------------------------------------------------------
# 5. İSİM ÜRETME (SAMPLING / GENERATION)
# ---------------------------------------------------------------------------
print("\n--- ÖRNEK İSİM ÜRETİMİ ---")
g_sample = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size 
    while True:
        emb = C[torch.tensor([context])] 
        embcat = emb.view(1, -1)
        # Yine kalibre edilmiş BatchNorm istatistiklerini kullanıyoruz
        hpreact = bngain * (embcat @ W1 + b1 - bnmean_calib) * (bnvar_calib + 1e-5)**-0.5 + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g_sample).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
            
    print(''.join(itos[i] for i in out))