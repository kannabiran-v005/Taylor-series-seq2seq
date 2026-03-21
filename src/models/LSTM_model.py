import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
torch.manual_seed(42)

import json
from torch.utils.data import Dataset, DataLoader

import random
random.seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"# hoping CUDA works, CPU is too slow
print(DEVICE)
class TaylorDataset(Dataset):
    def __init__(self, path):
        self.data = []
        
        # Tried loading all at once, but JSONL format requires streaming
        # This works fine
        
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

for x,y in dataset:
    token_set.update(x)
    token_set.update(y)

vocab = special + sorted(token_set)

stoi = {t:i for i,t in enumerate(vocab)}
itos = {i:t for t,i in stoi.items()}

PAD = stoi["<PAD>"]
SOS = stoi["<SOS>"]
EOS = stoi["<EOS>"]
print("Vocabulary size:", len(vocab)) # 1582 tokens, manageable
def encode(seq):
    return [stoi.get(t,stoi["<UNK>"]) for t in seq]
encoded = []

for x,y in dataset:
    src = encode(x)
    tgt = encode(y)

    encoded.append((src,tgt))
random.shuffle(encoded)
n = len(encoded)
train_end = int(0.8*n)
val_end = int(0.9*n)
train_data = encoded[:train_end]
val_data = encoded[train_end:val_end]
test_data = encoded[val_end:]
print("Train:",len(train_data))
print("Validation:",len(val_data))
print("Test:",len(test_data))
def collate(batch):

    src_batch = []
    tgt_in = []
    tgt_out = []

    max_src = max(len(x) for x,_ in batch)
    max_tgt = max(len(y) for _,y in batch)+1

    for src,tgt in batch:
        # Pad source
        src_pad = src + [PAD]*(max_src-len(src))
        # Decoder input = SOS + target sequence
        decoder_in = [SOS] + tgt
        # Decoder output = target + EOS
        decoder_out = tgt + [EOS]

        decoder_in += [PAD]*(max_tgt-len(decoder_in))
        decoder_out += [PAD]*(max_tgt-len(decoder_out))

        src_batch.append(src_pad)
        tgt_in.append(decoder_in)
        tgt_out.append(decoder_out)

    return (
        torch.tensor(src_batch),
        torch.tensor(tgt_in),
        torch.tensor(tgt_out)
    )

# Batch size 32 worked best — tried 64, got OOM errors
train_loader = DataLoader(train_data,batch_size=32,shuffle=True,collate_fn=collate)
val_loader = DataLoader(val_data,batch_size=32,shuffle=False,collate_fn=collate)
test_loader = DataLoader(test_data,batch_size=32,shuffle=False,collate_fn=collate)

class Seq2Seq(nn.Module):

    def __init__(self,vocab_size,emb=128,hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,emb,padding_idx=PAD)
        self.encoder = nn.LSTM(emb, hidden, num_layers=1,batch_first=True)
        self.decoder = nn.LSTM(emb, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden,vocab_size)

    def forward(self,src,tgt):
        src = self.embedding(src)
        _,(h,c) = self.encoder(src) # Don't need encoder outputs, just hidden state
        tgt = self.embedding(tgt)
        out,_ = self.decoder(tgt,(h,c))
        logits = self.fc(out)
        return logits
model = Seq2Seq(len(vocab)).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD)

# Learning rate: tried 1e-2 (diverged), 1e-4 (too slow), 1e-3 seems sweet spot
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

def evaluate(loader):

    model.eval()

    total_loss = 0
    total_tokens = 0
    correct = 0

    with torch.no_grad():

        for src,tin,tout in loader:

            src = src.to(DEVICE)
            tin = tin.to(DEVICE)
            tout = tout.to(DEVICE)

            logits = model(src,tin)

            loss = criterion(
                logits.view(-1,logits.size(-1)),
                tout.view(-1)
            )

            total_loss += loss.item()

            pred = logits.argmax(-1)

            mask = tout!=PAD

            correct += ((pred==tout) & mask).sum().item()
            total_tokens += mask.sum().item()

    acc = correct/total_tokens
    return total_loss/len(loader),acc
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(1,29):

    model.train()

    total_loss = 0
    total_tokens = 0
    correct = 0

    for src,tin,tout in train_loader:

        src = src.to(DEVICE)
        tin = tin.to(DEVICE)
        tout = tout.to(DEVICE)

        logits = model(src,tin)

        loss = criterion(
            logits.view(-1,logits.size(-1)),
            tout.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        # No gradient clipping yet — maybe add if gradients explode later
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        
        optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(-1)

        mask = tout!=PAD

        correct += ((pred==tout) & mask).sum().item()
        total_tokens += mask.sum().item()

    train_acc = correct/total_tokens
    train_loss = total_loss/len(train_loader)

    val_loss,val_acc = evaluate(val_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print("Epoch",epoch,"Train Loss:",train_loss,"Train Acc:",train_acc,"Val Loss:",val_loss, "Val Acc:",val_acc)

test_loss,test_acc = evaluate(test_loader)

print("\nFinal Test Results")
print("Test Loss:",test_loss)
print("Test Accuracy:",test_acc)

# Graphs 
epochs = list(range(1,len(train_losses)+1))
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs,train_losses,label="Train Loss")
plt.plot(epochs,val_losses,label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs,train_accs,label="Train Acc")
plt.plot(epochs,val_accs,label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("lstm_training_curves.png", dpi=300, bbox_inches="tight")
plt.show()
torch.save(model.state_dict(), "lstm_taylor_model.pth")
print("Model saved as lstm_taylor_model.pth")
