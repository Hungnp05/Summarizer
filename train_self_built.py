import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer

from app.self_built_model import Seq2SeqTransformer, tokenizer, DEVICE

# Config
BATCH_SIZE = 16  # Nhỏ hơn vì summarization nặng
EPOCHS = 3       # Để nhanh, tăng nếu cần
LEARNING_RATE = 1e-4
MAX_SRC_LEN = 512
MAX_TGT_LEN = 128

# Dataset CNN/DailyMail
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset['train'].select(range(5000))  # Subset để nhanh

class SummDataset(Dataset):
    def __init__(self, data):
        self.articles = data['article']
        self.summaries = data['highlights']

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = "summarize: " + self.articles[idx]
        summary = self.summaries[idx]
        src = tokenizer(article, max_length=MAX_SRC_LEN, truncation=True, padding='max_length', return_tensors='pt')
        tgt = tokenizer(summary, max_length=MAX_TGT_LEN, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': src['input_ids'].squeeze(),
            'labels': tgt['input_ids'].squeeze()
        }

train_dataset = SummDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Seq2SeqTransformer(vocab_size=tokenizer.vocab_size)
model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

def train_epoch():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids, labels[:, :-1])  # Shifted target
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels[:, 1:].view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        train_loss = train_epoch()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")
    torch.save(model.state_dict(), "summarizer_self_built.pt")
    print("Đã lưu model.")