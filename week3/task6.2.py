import torch
import torch.nn.functional as F

isimler = open('week3/isimler.txt', encoding='utf-8').read().splitlines()
isimler = [isim.replace(' ', '-') for isim in isimler]
chars = sorted(list(set(''.join(isimler))))
chars.append('.')
htos = {s: i for i, s in enumerate(chars)}
stoh = {i: s for s, i in htos.items()}

W = torch.load('week3/trigram_agirliklari.pt')
g = torch.Generator().manual_seed(2147483647)


try:
    isim_sayisi = int(input("Kaç adet isim üretilsin? "))
except ValueError:
    print("Geçersiz sayı girildi, varsayılan olarak 10 isim üretilecek.")
    isim_sayisi = 10  


print("\nNasıl bir üretim yapmak istersin?")
print("1 - Tamamen rastgele üret (Harf seçimi yok)")
print("2 - Sadece ilk harfi ben seçeyim")
print("3 - İlk iki harfi ben seçeyim")

secim = input("Seçiminiz (1/2/3): ")


baslangic = ""
if secim == '2':
    while True:
        baslangic = input("İsmin ilk harfini girin (Örn: A): ").upper()
        if len(baslangic) == 1 and baslangic in htos:
            break
        print("Hata: Lütfen geçerli 1 Türkçe harf girin.")
elif secim == '3':
    while True:
        baslangic = input("İsmin ilk iki harfini girin (Örn: AL): ").upper()
        if len(baslangic) == 2 and baslangic[0] in htos and baslangic[1] in htos:
            break
        print("Hata: Lütfen geçerli 2 Türkçe harf girin.")



print('------------------')
print('NETWORK AĞI İLE TRİGRAM İSİM ÜRETİMİ')
print('------------------')



# 4. Üretim Döngüsü
for i in range(isim_sayisi):
    
    # Kullanıcının seçimine göre başlangıç durumunu ayarla
    if secim == '1' or secim not in ['2', '3']:
        ix1, ix2 = 30, 30
        out = []
    elif secim == '2':
        # Sadece 1 harf biliniyorsa, ondan önceki harf mecburen başlangıç noktasıdır (30)
        ix1, ix2 = 30, htos[baslangic]
        out = [baslangic]
    elif secim == '3':
        ix1, ix2 = htos[baslangic[0]], htos[baslangic[1]]
        out = [baslangic]
        
    while True:
        kombine_ix = ix1 * 31 + ix2
        xenc = F.one_hot(torch.tensor([kombine_ix]), num_classes=961).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        
        ix3 = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(stoh[ix3])
        
        if ix3 == 30:
            break
        
        ix1 = ix2
        ix2 = ix3
        
    print(''.join(out))
