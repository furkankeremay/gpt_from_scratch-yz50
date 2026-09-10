import torch
import torch.nn.functional as F



words = (open('week4/names.txt').read().splitlines())
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

g = torch.Generator().manual_seed(2147483647)


#OLD BIGRAM MODEL

W = torch.load('week3/bigram_agirliklari.pt')

bigram_isimler = []

for i in range(10):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        
        if ix == 0:
            break
    bigram_isimler.append(''.join(out))




#NEW MODEL

block_size = 3

agirliklar = torch.load('week4/mlp_agirliklari.pt')
C = agirliklar['C']
W1 = agirliklar['W1']
b1 = agirliklar['b1']
W2 = agirliklar['W2']
b2 = agirliklar['b2']

mlp_isimler = []

for _ in range(10):
    out = []
    context = [0] * block_size
    
    while True:
        emb = C[torch.tensor([context])] # Boyut: (1, 3, 10)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1) # 1x30 matrisi ile gizli katmana giriş
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        
        context = context[1:] + [ix]
        out.append(itos[ix])
        
        # Kelime bitiş karakteri üretildiyse kelimeyi sonlandır
        if ix == 0:
            break
            
    mlp_isimler.append(''.join(out))

print(f"{'ESKİ BİGRAM MODELİ':<30} | {'YENİ MLP MODELİ'}")
print("-" * 55)

for bi_isim, mlp_isim in zip(bigram_isimler, mlp_isimler):
    print(f"{bi_isim:<30} | {mlp_isim}")