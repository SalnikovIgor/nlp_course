import datasets
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


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


def init():
    torch.manual_seed(123)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = "google/bert_uncased_L-4_H-256_A-4"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=4, ignore_mismatched_sizes=True)

    model = model.to(device)

    dataset = datasets.load_dataset('imdb')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256,
                         return_tensors='pt').to(device)

    train = dataset['train'].map(tokenize_function, batched=True)
    val = dataset['test'].map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=64
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train
    )
    trainer.train()

    labels = val['label']
    clear_val = val.map(remove_columns=['text', 'label'])
    clear_val.set_format('pt')

    print(clear_val)

    val_loader = DataLoader(clear_val, batch_size=64)

    predict = []
    for batch in val_loader:
        output = model(input_ids=batch['input_ids'].to(device),
                       token_type_ids=batch['token_type_ids'].to(device),
                       attention_mask=batch['attention_mask'].to(device))
        predict.extend(torch.argmax(output.logits, dim=1))

    acc = evaluate.load('accuracy')

    print(acc.compute(predictions=predict, references=labels))
