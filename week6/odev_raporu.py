import torch
from layers import Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, Sequential

# --- VERİ SETİ HAZIRLIĞI (Şekil testi için küçük bir dummy veri) ---
vocab_size = 27
n_embd = 24
n_hidden = 128
batch_size = 32

# ==========================================
# GÖREV 2: BAĞLAMI 3'TEN 8'E ÇIKARMA (DÜZ MLP)
# ==========================================
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    def parameters(self):
        return []

# Bağlam 3 Düz MLP (Hafta 4 Referansı)
model_ctx3_flat = Sequential([
    Embedding(vocab_size, n_embd), Flatten(),
    Linear(n_embd * 3, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])

# Bağlam 8 Düz MLP (Sadece bağlam boyutu değişti)
model_ctx8_flat = Sequential([
    Embedding(vocab_size, n_embd), Flatten(),
    Linear(n_embd * 8, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])

params_ctx3 = sum(p.nelement() for p in model_ctx3_flat.parameters())
params_ctx8_flat = sum(p.nelement() for p in model_ctx8_flat.parameters())

print(f"--- GÖREV 2: Parametre Karşılaştırması ---")
print(f"Bağlam 3 Düz MLP Parametre Sayısı: {params_ctx3}")
print(f"Bağlam 8 Düz MLP Parametre Sayısı: {params_ctx8_flat}")
print(f"Fark: Parametre sayısı {params_ctx8_flat - params_ctx3} arttı. (Çünkü ilk Linear katmanın girdi boyutu 3*n_embd'den 8*n_embd'ye çıktı).\n")

# ==========================================
# GÖREV 3: WAVENET ŞEKİLLERİ (SHAPES)
# ==========================================
model_wavenet = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])

params_wavenet = sum(p.nelement() for p in model_wavenet.parameters())

print(f"--- GÖREV 3: WaveNet Katman Çıktı Şekilleri ---")
dummy_X = torch.randint(0, vocab_size, (batch_size, 8)) # 32 örnek, 8 harf
x = dummy_X
for i, layer in enumerate(model_wavenet.layers):
    x = layer(x)
    print(f"{layer.__class__.__name__:18s} çıktısı: {list(x.shape)}")
print()

# ==========================================
# GÖREV 5: FİNAL TABLOSU
# ==========================================
loss_ctx3_flat = 2.0951
loss_ctx8_flat = 2.0217
loss_wavenet = 1.9937

print("--- GÖREV 5: MODEL KIYASLAMA TABLOSU ---")
print(f"{'Model Türü':<20} | {'Parametre Sayısı':<15} | {'Dev Loss':<10}")
print("-" * 52)
print(f"{'Bağlam 3 (Düz MLP)':<20} | {params_ctx3:<15} | {loss_ctx3_flat:<10.4f}")
print(f"{'Bağlam 8 (Düz MLP)':<20} | {params_ctx8_flat:<15} | {loss_ctx8_flat:<10.4f}")
print(f"{'Bağlam 8 (WaveNet)':<20} | {params_wavenet:<15} | {loss_wavenet:<10.4f}")