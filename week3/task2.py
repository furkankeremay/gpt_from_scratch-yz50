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



#a = sorted(b.items(), key=lambda kv: -kv[1])

#p = N[0, :].float()
#p = p / p.sum()

g=torch.Generator().manual_seed(2147483647)

P = N.float()
P /= P.sum(1, keepdim=True)

for i in range(5):
    ix = 0
    out = []

    while (True):

        p = P[ix, :]
        #p = N[ix, :].float()
        #p = p / p.sum()
        #p = torch.ones(27 ) / 27
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
