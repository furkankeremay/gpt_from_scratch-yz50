import torch
import matplotlib.pyplot as plt

words = (open('week3/names.txt').read().splitlines())
b = {}


N = torch.zeros((27,27), dtype =torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    w = ['.'] +  list(w) + ['.']
    for ch1,ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

        bigram = (ch1 , ch2)
        b[bigram] = b.get(bigram, 0) + 1


plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');


a = sorted(b.items(), key=lambda kv: -kv[1])
print(a)
plt.savefig('bigram_grafigi_eng.png')