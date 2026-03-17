import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json
import random
from torch.utils.data import Dataset, DataLoader

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
torch.manual_seed(42)

random.seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

class TaylorDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.data.append((obj["in_tokens"], obj["out_tokens"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = TaylorDataset("taylor_tokenized_dataset.jsonl")
print("Total samples:", len(dataset))

special = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
token_set = set()

for x, y in dataset:
    token_set.update(x)
    token_set.update(y)

vocab = special + sorted(token_set)

stoi = {t: i for i, t in enumerate(vocab)}
itos = {i: t for t, i in stoi.items()}

PAD = stoi["<PAD>"]
SOS = stoi["<SOS>"]
EOS = stoi["<EOS>"]

print("Vocabulary size:", len(vocab))
def encode(seq):
    return [stoi.get(t, stoi["<UNK>"]) for t in seq]

encoded = []
for x, y in dataset:
    src = encode(x)
    tgt = encode(y)
    encoded.append((src, tgt))

random.shuffle(encoded)
n = len(encoded)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_data = encoded[:train_end]
val_data = encoded[train_end:val_end]
test_data = encoded[val_end:]

print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))

def collate(batch):
    src_batch = []
    tgt_in = []
    tgt_out = []

    max_src = max(len(x) for x, _ in batch)
    max_tgt = max(len(y) for _, y in batch) + 1

    for src, tgt in batch:
        src_pad = src + [PAD] * (max_src - len(src))

        decoder_in = [SOS] + tgt
        decoder_out = tgt + [EOS]

        decoder_in += [PAD] * (max_tgt - len(decoder_in))
        decoder_out += [PAD] * (max_tgt - len(decoder_out))

        src_batch.append(src_pad)
        tgt_in.append(decoder_in)
        tgt_out.append(decoder_out)

    return (
        torch.tensor(src_batch, dtype=torch.long),
        torch.tensor(tgt_in, dtype=torch.long),
        torch.tensor(tgt_out, dtype=torch.long)
    )

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, emb=128, nhead=8, num_layers=4, ff_dim=512, dropout=0.05):
        super().__init__()

        self.emb = emb
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.pos_encoder = PositionalEncoding(emb)

        self.transformer = nn.Transformer(
            d_model=emb,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(emb, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        src_pad_mask = (src == PAD)
        tgt_pad_mask = (tgt == PAD)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

        src = self.embedding(src) * math.sqrt(self.emb)
        tgt = self.embedding(tgt) * math.sqrt(self.emb)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        logits = self.fc(out)
        return logits

model = TransformerSeq2Seq(
    len(vocab),
    emb=128,
    nhead=8,
    num_layers=4,
    ff_dim=512,
    dropout=0.05
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # reduce when val loss stops decreasing
    factor=0.5,        # LR becomes half
    patience=2,        # wait 2 epochs before reducing
)


def evaluate(loader):
    model.eval()

    total_loss = 0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for src, tin, tout in loader:
            src = src.to(DEVICE)
            tin = tin.to(DEVICE)
            tout = tout.to(DEVICE)

            logits = model(src, tin)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tout.view(-1)
            )

            total_loss += loss.item()

            pred = logits.argmax(-1)
            mask = tout != PAD

            correct += ((pred == tout) & mask).sum().item()
            total_tokens += mask.sum().item()

    acc = correct / total_tokens
    return total_loss / len(loader), acc

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(1, 38):
    model.train()

    total_loss = 0
    total_tokens = 0
    correct = 0

    for src, tin, tout in train_loader:
        src = src.to(DEVICE)
        tin = tin.to(DEVICE)
        tout = tout.to(DEVICE)

        logits = model(src, tin)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            tout.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(-1)
        mask = tout != PAD

        correct += ((pred == tout) & mask).sum().item()
        total_tokens += mask.sum().item()

    train_acc = correct / total_tokens
    train_loss = total_loss / len(train_loader)

    val_loss, val_acc = evaluate(val_loader)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print("Epoch", epoch,"Train Loss:", train_loss,"Train Acc:", train_acc,"Val Loss:", val_loss,"Val Acc:", val_acc)
test_loss, test_acc = evaluate(test_loader)

print("\nFinal Test Results")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

epochs = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Loss vs Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transformer Accuracy vs Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("transformer_training_curves.png", dpi=300, bbox_inches="tight")
plt.show()

torch.save(model.state_dict(), "transformer_taylor_model.pth")
print("Model saved as transformer_taylor_model.pth")