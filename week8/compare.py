import json

kayit_yolu = "week8/week7_loss_sonuclari.json"
with open(kayit_yolu, "r", encoding="utf-8") as f:
    loss_verileri = json.load(f)

kayit_yolu = "week8/week8_mhead_losses.json"
with open(kayit_yolu, "r", encoding="utf-8") as f:
    losses = json.load(f)

kayit_yolu = "week8/week8_gpt_losses.json"
with open(kayit_yolu, "r", encoding="utf-8") as f:
    gpt_losses = json.load(f)

print(f'{"Tek head train loss":<20}: {loss_verileri["train_loss"]:.4f}, val loss: {loss_verileri["val_loss"]:.4f}')
print(f'{"Çok head train loss":<20}: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')
print(f'{"GPT train loss":<20}: {gpt_losses["train"]:.4f}, val loss: {gpt_losses["val"]:.4f}')