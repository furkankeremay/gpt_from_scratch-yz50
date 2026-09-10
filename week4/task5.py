import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

names = open('week4/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(names))))
stoi = {'.': 0, **{s:i+1 for i,s in enumerate(chars)}}

def build_dataset(dataset):
    X, Y = [], []
    for isim in dataset:
        context = [0] * 3
        for ch in isim + '.':
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(names)
Xtr, Ytr = build_dataset(names[:int(len(names)*0.8)])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)

W1_bad = torch.randn((30, 300), generator=g)
b1_bad = torch.randn(300, generator=g)
W2_bad = torch.randn((300, 27), generator=g)
b2_bad = torch.randn(27, generator=g)

ix = torch.randint(0, Xtr.shape[0], (32,))
emb = C[Xtr[ix]]

hpreact_bad = emb.view(-1, 30) @ W1_bad + b1_bad
h_bad = torch.tanh(hpreact_bad)
logits_bad = h_bad @ W2_bad + b2_bad
loss_bad = F.cross_entropy(logits_bad, Ytr[ix])

kaiming_scale = (5.0/3.0) / (30 ** 0.5)
W1_good = torch.randn((30, 300), generator=g) * kaiming_scale
b1_good = torch.randn(300, generator=g) * 0.01
W2_good = torch.randn((300, 27), generator=g) * 0.01
b2_good = torch.randn(27, generator=g) * 0.0

hpreact_good = emb.view(-1, 30) @ W1_good + b1_good
h_good = torch.tanh(hpreact_good)
logits_good = h_good @ W2_good + b2_good
loss_good = F.cross_entropy(logits_good, Ytr[ix])

print(f"İdeal Kayıp: {-torch.tensor(1/27.0).log().item():.4f}")
print(f"Sorunlu Kayıp: {loss_bad.item():.4f}")
print(f"Düzeltilmiş Kayıp: {loss_good.item():.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(h_bad.view(-1).tolist(), bins=50, color='red', alpha=0.7)
plt.title('Sorunlu Tanh')
plt.subplot(1, 2, 2)
plt.hist(h_good.view(-1).tolist(), bins=50, color='green', alpha=0.7)
plt.title('Kaiming Init')
plt.show()