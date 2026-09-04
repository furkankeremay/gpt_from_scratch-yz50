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



g=torch.Generator().manual_seed(2147483647)

P = (N+1).float()
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

log_likelihood = 0.0
n=0

for w in words:
    w = ['.'] +  list(w) + ['.']
    for ch1,ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        n += 1
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob

#print(f'log likelihood: {log_likelihood:.4f}')
#print(f'negative log likelihood: {-log_likelihood:.4f}')
print(f'average negative log likelihood: {-log_likelihood/n:.4f}')