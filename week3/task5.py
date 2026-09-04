import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

isimler =open('week3/isimler.txt', encoding = 'utf-8').read().splitlines()

isimler = [isim.replace(' ', '-') for isim in isimler]

N = torch.zeros((31,31), dtype =torch.int32)


chars = sorted(list(set(''.join(isimler))))
for i in range(len(chars)-1):
    chars[i] = chars[i+1]
chars.pop(29)
chars.append('.')
chars.insert(0, '-')
#print(chars)
htos = {s:i for i,s in enumerate(chars)}
stoh = {i:s for s,i in htos.items()}
#print(htos, stoh)

xs,ys = [],[]

for isim in isimler:
    isim = ['.'] +  list(isim) + ['.']
    for hf1, hf2 in zip(isim, isim[1:]):
        ix1 = htos[hf1]
        ix2 = htos[hf2]
        N[ix1, ix2] += 1
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

P = (N+1).float()
P /= P.sum(1, keepdim=True)

'''
plt.figure(figsize=(31,31))
plt.imshow(P, cmap='Blues')
for i in range(31):
    for j in range(31):
        chstr = stoh[i] + stoh[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontsize=18)

        prob = P[i, j].item()
        plt.text(j, i, f'{prob:.2f}', ha="center", va="top", color='gray', fontsize=18)
plt.axis('off');



plt.savefig('bigram_grafigi_olasılık.png')

'''

n = 0
log_likelihood = 0.0

for w in isimler:
    w = ['.'] +  list(w) + ['.']
    for ch1,ch2 in zip(w, w[1:]):
        ix1 = htos[ch1]
        ix2 = htos[ch2]
        n += 1
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob



g=torch.Generator().manual_seed(2147483647)

print('------------------')
print('BİGRAM İSİM ÜRETİMİ')
print('------------------')
for i in range(10):
    ix = 30
    out = []

    while (True):

        p = P[ix, :]
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(stoh[ix])
        if ix == 30:
            break
    print(''.join(out))

print('------------------')
print(f'ORTALAMA NEGATİVE LOGLİKELİHOOD ( LOSS ): {-log_likelihood/n:.4f}')
print('------------------')



g = torch.Generator().manual_seed(2147483647)
W = torch.randn((31, 31), requires_grad=True, generator=g)

print('------------------')
print('NETWORK AĞI İLE LOSS AZALTMA')
print('------------------')

for i in range (401): 
    xenc = F.one_hot(xs, num_classes=31).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

    W.grad = None
    loss.backward()
    W.data += -100 * W.grad
    if i < 395:
        if i % 50 == 0 or i > 395 == 0:
            print(f'{i}. deneme: loss = {loss.item():.4f}')
    else:
        print(f'{i}. deneme: loss = {loss.item():.4f}')

print('------------------')
print('NETWORK AĞI İLE İSİM ÜRETİMİ')
print('------------------')

for i in range(10):
    out = []
    ix = 30
    while (True):
        xenc = F.one_hot(torch.tensor([ix]), num_classes=31).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(stoh[ix])
        if ix == 30:
            break
    print(''.join(out))






