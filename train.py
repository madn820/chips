import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tinygpt_model import TinyGPT

# تنظیمات
BATCH_SIZE = 16
SEQ_LEN = 32
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# لود توکنایزر
tokenizer = Tokenizer.from_file("tinygpt_tokenizer.json")

# دیتاست سفارشی
class TextDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text).ids
        self.data = [tokens[i:i+SEQ_LEN] for i in range(0, len(tokens) - SEQ_LEN)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:] + [0], dtype=torch.long)
        return x, y

# ساخت DataLoader
dataset = TextDataset("dataset.txt")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ساخت مدل
vocab_size = tokenizer.get_vocab_size()
model = TinyGPT(vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

# آموزش مدل
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in loader:
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(loader):.4f}")

# ذخیره مدل
torch.save(model.state_dict(), "tinygpt_model.pth")
print("✅ مدل آموزش دیده و ذخیره شد.")
