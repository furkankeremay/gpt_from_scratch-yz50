import torch
import torch.nn.functional as F
import random


isimler = (open('week3/isimler.txt', encoding='utf-8').read().splitlines())
isimler = [isim.replace(' ', '-') for isim in isimler]

chars = sorted(list(set(''.join(isimler))))
chars.append('.')
htos = {s:i for i,s in enumerate(chars)}
stoh = {i:s for s,i in htos.items()}

print(htos)


random.seed(42)
random.shuffle(isimler)

n1 = int(len(isimler) * 0.8)
n2 = int(len(isimler) * 0.9)

train_isimler = isimler[:n1]
dev_isimler = isimler[n1:n2]
test_isimler = isimler[n2:]

def veri_seti_olustur(kelime_listesi):
    X, Y = [], []
    for w in kelime_listesi:
        w = ['.'] + ['.'] + list(w) + ['.']
        for hf1, hf2, hf3 in zip(w, w[1:], w[2:]):
            ix1 = htos[hf1]
            ix2 = htos[hf2]
            ix3 = htos[hf3]
            çarpım = 31*ix1+ix2
            X.append(çarpım)
            Y.append(ix3)
    return torch.tensor(X), torch.tensor(Y)

X_train, Y_train = veri_seti_olustur(train_isimler)
X_dev, Y_dev = veri_seti_olustur(dev_isimler)
X_test, Y_test = veri_seti_olustur(test_isimler)
num_train = X_train.nelement()

g = torch.Generator().manual_seed(2147483647)

W = torch.randn((961,31), requires_grad=True, generator=g)

print('------------------')
print('TRAIN SETİ İLE EĞİTİM (LOSS AZALTMA)')
print('------------------')


for i in range(401):
    xenc = F.one_hot(X_train, num_classes=961).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num_train), Y_train].log().mean() +  0.01*(W**2).mean()

    W.grad = None
    loss.backward()
    W.data += -100 * W.grad

    if i<395:
        if i % 50 == 0:
            print(f'{i}. deneme: loss = {loss.item():.4f}')
    else:
        print(f'{i}. deneme: loss = {loss.item():.4f}')


torch.save(W, 'trigram_agirliklari.pt')
print("Eğitim tamamlandı. Ağırlıklar 'trigram_agirliklari.pt' olarak kaydedildi.")



