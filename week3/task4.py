import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = (open('week3/names.txt').read().splitlines())

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


xs,ys = [],[]


for w in words:
    w = ['.'] +  list(w) + ['.']
    for ch1,ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), requires_grad=True, generator=g)


for k in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    #print(f'{k}. attempt: loss = {loss.item():.4f}')

    W.grad = None

    loss.backward()

    W.data += -50 * W.grad


for i in range(5):
    out = []
    ix = 0
    while (True):
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
print(f'loss: {loss.item():.4f}')


