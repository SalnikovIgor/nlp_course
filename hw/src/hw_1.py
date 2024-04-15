import datasets
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, device='cuda'):
        self.encodings = encodings
        self.labels = labels
        self.device = device

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(self.device)
        return item

    def __len__(self):
        return len(self.labels)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, vocab_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_dim)
        self.rnn = nn.LSTM(input_size=vocab_dim, hidden_size=vocab_dim,
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


def evaluate(model, val_loader, epoch, device):
    print(model)
    model.eval()
    predictions = []
    target = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} test: "):
            logits = model(batch["input_ids"].to(device))
            predictions.append(logits.argmax(dim=1))
            target.append(batch['labels'])

    predictions = torch.cat(predictions)
    target = torch.cat(target)
    accuracy = (predictions == target).float().mean().item()
    print()
    print(accuracy)


def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, scheduler, device):
    for epoch in tqdm(range(epochs)):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} train: "):
            optimizer.zero_grad()
            xb, yb = batch["input_ids"], batch["labels"]
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()

        evaluate(model, val_loader, epoch, device)


def init():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Загрузим датасет и разобьем его на train и val subsets
    dataset = datasets.load_dataset('ag_news')
    dataset = dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataset['train']['text'], dataset['train']['label'], test_size=.1)

    # 2. Токенезируем subsets
    seq_size = 256
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=seq_size)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=seq_size)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)

    batch_size = 64
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size)

    eval_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size)

    model = LanguageModel(len(tokenizer.vocab), seq_size).to(device)

    epochs = 5
    lr = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    train_model(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs, scheduler, device)
