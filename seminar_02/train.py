from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Загрузим датасет и разобьем его на train и val subsets
dataset = datasets.load_dataset('ag_news')
dataset = dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset['train']['text'], dataset['train']['label'], test_size=.1)

# 2. Токенезируем subsets
seq_size = 100
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=seq_size)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=seq_size)


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

batch_size = 256
train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size)

eval_dataloader = DataLoader(
    val_dataset, shuffle=False, batch_size=batch_size)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, vocab_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_dim)
        self.rnn = nn.RNN(input_size=vocab_dim, hidden_size=vocab_dim,
                          num_layers=2, batch_first=True, bidirectional=False)
        self.fc_1 = nn.Linear(vocab_dim, vocab_dim)
        self.fc_2 = nn.Linear(vocab_dim, 4)

    def forward(self, x):
        embedding = self.embedding(x)
        x, _ = self.rnn(embedding)
        x = x.mean(dim=1)
        x = torch.tanh(x)
        x = self.fc_1(x)
        x = torch.tanh(x)
        x = self.fc_2(x)
        return x


model = LanguageModel(len(tokenizer.vocab), seq_size).to(device)

epochs = 6
lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.8, gamma=0.125)


def evaluate(model, val_loader, epoch):
    model.eval()
    predictions = []
    target = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} test: "):
            xb = batch["input_ids"]
            logits = model(xb.to(device))
            predictions.append(logits.argmax(dim=1))
            target.append(batch['labels'].to(device))

    predictions = torch.cat(predictions)
    target = torch.cat(target)
    accuracy = (predictions == target).float().mean().item()
    print()
    print(accuracy)


def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, scheduler):
    for epoch in tqdm(range(epochs)):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} train: "):
            optimizer.zero_grad()
            xb, yb = batch["input_ids"], batch["labels"]
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        evaluate(model, val_loader, epoch)


train_model(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs, scheduler)
